# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from vllm.v1.worker.gpu import pp_utils


def _batch(
    num_computed,
    prefill_len,
    num_scheduled,
    is_prefilling=None,
    max_seq_len=None,
):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
        is_prefilling_np=(
            np.array(is_prefilling, dtype=np.bool_)
            if is_prefilling is not None
            else None
        ),
        max_seq_len_np=(
            np.array(max_seq_len, dtype=np.int32) if max_seq_len is not None else None
        ),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


def test_excludes_final_prefill_chunk_reaching_its_length_cap():
    """A max_tokens=1 final chunk hands its token off and leaves the engine.

    Disaggregated prefill submits max_tokens=1, so the request finishes with
    the final chunk's single sample and no follow-up step on this engine
    reads the broadcast payload.
    """
    batch = _batch(
        num_computed=[1000],
        prefill_len=[1004],
        num_scheduled=[4],
        is_prefilling=[True],
        max_seq_len=[1005],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_final_prefill_chunk_with_generation_budget_left():
    """The same final chunk, but the request keeps decoding here."""
    batch = _batch(
        num_computed=[1000],
        prefill_len=[1004],
        num_scheduled=[4],
        is_prefilling=[True],
        max_seq_len=[1004 + 256],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_length_cap_overrun_decode_row_is_kept():
    """Decoding rows are never excluded, even past their length cap.

    Speculative decoding can transiently overrun prompt_len + max_tokens
    while the scheduler still runs the request; dropping the row would
    desynchronize the PP stages.
    """
    batch = _batch(
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
        is_prefilling=[False],
        max_seq_len=[12176],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_finishing_prefill_row_does_not_drop_a_sibling_row():
    """Only finishing rows are excluded; a decoding sibling keeps its row."""
    batch = _batch(
        num_computed=[1000, 10],
        prefill_len=[1004, 8],
        num_scheduled=[4, 1],
        is_prefilling=[True, False],
        max_seq_len=[1005, 1024],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_draft_broadcast_survives_proposer_buffer_reuse(monkeypatch):
    """Snapshot selected request slots before entering the asynchronous stream."""
    handler = object.__new__(pp_utils.PPHandler)
    handler.is_last_rank = True
    handler.last_rank = 1
    handler.main_stream = Mock()
    handler.broadcast_stream = Mock()
    handler.broadcast_group = None
    batch = SimpleNamespace(idx_mapping=torch.tensor([3, 1]))
    drafts = torch.arange(12, dtype=torch.int64).view(4, 3)
    expected = drafts[batch.idx_mapping].clone()
    received = []

    @contextmanager
    def delayed_stream(stream):
        drafts.fill_(99)
        yield

    monkeypatch.setattr(pp_utils, "compute_need_sampled_mask", lambda _: [True])
    monkeypatch.setattr(torch.cuda, "stream", delayed_stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, **kw: received.append(tensor)
    )
    handler.broadcast_drafts(drafts, batch)
    torch.testing.assert_close(received[0], expected)
    handler.broadcast_stream.wait_stream.assert_called_once_with(handler.main_stream)


def test_deferred_drafts_skip_reused_request_slot(monkeypatch):
    """A cancelled slot must retain its new request's draft proposals."""
    handler = object.__new__(pp_utils.PPHandler)
    handler.device = torch.device("cpu")
    handler.main_stream = Mock()
    handler.req_idx_gen_np = np.zeros(4, dtype=np.int32)
    mapping_np = np.array([3, 1], dtype=np.int32)
    drafts = torch.tensor([[10, 11, 12], [20, 21, 22]])
    handler.queue = deque(
        [
            pp_utils.PendingRecv(
                event=Mock(),
                sampled_tokens=torch.zeros(2, 4, dtype=torch.int64),
                num_sampled=torch.ones(2, dtype=torch.int32),
                num_rejected=torch.zeros(2, dtype=torch.int32),
                idx_mapping=torch.tensor(mapping_np),
                idx_mapping_np=mapping_np,
                need_sampled_mask=np.ones(2, dtype=bool),
                gen_at_receive_np=handler.req_idx_gen_np[mapping_np].copy(),
                draft_tokens=drafts,
            )
        ]
    )
    monkeypatch.setattr(
        pp_utils, "async_tensor_h2d", lambda array, device: torch.tensor(array)
    )
    handler.on_req_idx_freed(1)
    destination = torch.full((4, 3), 99, dtype=torch.int64)
    expected = destination.clone()
    expected[3] = drafts[0]
    outputs = handler.get_prev_sampled_outputs(destination)
    assert outputs["idx_mapping"].tolist() == [3, -1]
    torch.testing.assert_close(destination, expected)
