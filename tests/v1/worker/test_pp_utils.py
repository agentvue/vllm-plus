# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.pp_utils import PendingRecv, PPHandler, scatter_draft_tokens

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_draft_broadcast_survives_proposer_buffer_reuse(monkeypatch):
    """A queued send must retain its tokens after the next proposal overwrites them."""
    handler = object.__new__(PPHandler)
    handler.is_last_rank = True
    handler.last_rank = 1
    handler.relay_draft_tokens = True
    handler.main_stream = torch.cuda.current_stream()
    handler.broadcast_stream = torch.cuda.Stream()
    handler.broadcast_group = None
    batch = SimpleNamespace(
        num_computed_tokens_np=np.array([10, 20]),
        prefill_len_np=np.array([10, 20]),
        num_scheduled_tokens=np.array([4, 4]),
        max_seq_len_np=np.array([100, 100]),
    )
    drafts = torch.arange(6, dtype=torch.int64, device="cuda").view(2, 3)
    expected = drafts.clone()
    pending = []

    # Defer collective consumption until after the proposer reuses its buffer.
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, **kwargs: pending.append(tensor)
    )
    handler.broadcast_draft(drafts, batch)
    drafts.fill_(99)
    handler.broadcast_stream.wait_stream(handler.main_stream)
    with torch.cuda.stream(handler.broadcast_stream):
        received = pending.pop().clone()
    handler.main_stream.wait_stream(handler.broadcast_stream)
    torch.testing.assert_close(received, expected)


def test_deferred_drafts_skip_reused_request_slot():
    """An old PP result cannot overwrite a new request occupying a freed slot."""
    handler = object.__new__(PPHandler)
    handler.device = torch.device("cuda")
    handler.main_stream = torch.cuda.current_stream()
    handler.req_idx_gen_np = np.zeros(4, dtype=np.int32)
    mapping_np = np.array([3, 1], dtype=np.int32)
    drafts = torch.tensor([[10, 11, 12], [20, 21, 22]], device="cuda")
    handler.queue = deque(
        [
            PendingRecv(
                event=handler.main_stream.record_event(),
                sampled_tokens=torch.zeros(2, 4, dtype=torch.int64, device="cuda"),
                num_sampled=torch.ones(2, dtype=torch.int32, device="cuda"),
                num_rejected=torch.zeros(2, dtype=torch.int32, device="cuda"),
                idx_mapping=torch.tensor(mapping_np, device="cuda"),
                idx_mapping_np=mapping_np,
                need_sampled_mask=np.ones(2, dtype=bool),
                gen_at_receive_np=handler.req_idx_gen_np[mapping_np].copy(),
                draft_tokens=drafts,
            )
        ]
    )
    handler.on_req_idx_freed(1)
    destination = torch.full((4, 3), 99, dtype=torch.int64, device="cuda")
    expected = destination.clone()
    expected[3] = drafts[0]

    outputs = handler.get_prev_sampled_outputs()
    assert outputs is not None
    scatter_draft_tokens(destination, outputs["draft_tokens"], outputs["idx_mapping"])

    torch.testing.assert_close(destination, expected)
