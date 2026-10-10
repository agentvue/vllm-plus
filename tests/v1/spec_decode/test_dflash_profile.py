# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class _DefaultSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):
        raise NotImplementedError


class _StopDummyRun(Exception):
    pass


def _run_dummy_batch(
    speculator,
    *,
    max_num_reqs=512,
    is_profile=False,
    skip_attn=False,
):
    captured = {}

    def execute_model(scheduler_output, **kwargs):
        captured["scheduler_output"] = scheduler_output
        raise _StopDummyRun

    runner = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        max_num_tokens=2048,
        decode_query_len=1,
        speculator=speculator,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
        is_first_pp_rank=True,
        lora_config=None,
        maybe_dummy_run_with_lora=lambda *args, **kwargs: nullcontext(),
        execute_model=execute_model,
    )

    with pytest.raises(_StopDummyRun):
        GPUModelRunner._dummy_run(
            runner,
            num_tokens=2048,
            skip_attn=skip_attn,
            is_profile=is_profile,
        )

    return captured["scheduler_output"]


def test_default_speculator_query_width_is_one():
    assert _DefaultSpeculator().num_query_per_req == 1


@pytest.mark.parametrize("speculator_cls", [DFlashSpeculator, DSparkSpeculator])
@pytest.mark.parametrize(
    ("is_profile", "skip_attn"),
    [(False, False), (True, False), (True, True)],
)
def test_dummy_run_caps_parallel_draft_requests(speculator_cls, is_profile, skip_attn):
    speculator = speculator_cls.__new__(speculator_cls)
    speculator.num_query_per_req = 5

    scheduler_output = _run_dummy_batch(
        speculator,
        is_profile=is_profile,
        skip_attn=skip_attn,
    )
    assert len(scheduler_output.num_scheduled_tokens) == 409
    assert scheduler_output.total_num_scheduled_tokens == 2048


def test_dummy_run_keeps_already_fitting_parallel_draft_batch():
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 5

    scheduler_output = _run_dummy_batch(speculator, max_num_reqs=128)
    assert len(scheduler_output.num_scheduled_tokens) == 128


def test_dummy_run_keeps_default_speculator_batch():
    scheduler_output = _run_dummy_batch(_DefaultSpeculator())
    assert len(scheduler_output.num_scheduled_tokens) == 512


def test_flashinfer_draft_workspace_is_reserved_during_profiling(monkeypatch):
    from vllm.v1.attention.backends import flashinfer
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    allocated = []
    monkeypatch.setattr(
        flashinfer, "_get_trtllm_workspace_buffer", lambda: allocated.append(True)
    )
    speculator = object.__new__(DFlashSpeculator)
    speculator.speculative_config = SimpleNamespace(
        attention_backend=AttentionBackendEnum.FLASHINFER
    )
    speculator._reserve_attention_workspace_for_profile()
    assert allocated == [True]


@pytest.mark.parametrize("has_own", [False, True])
def test_pp_dflash_loads_missing_embedding_from_pinned_target(monkeypatch, has_own):
    import torch

    from vllm.v1.worker.gpu.spec_decode.dflash import utils

    loaded = []
    param = torch.nn.Parameter(torch.zeros(2, 3))
    param.weight_loader = lambda param, weight: loaded.append(weight)
    draft = SimpleNamespace(has_own_embed_tokens=has_own)
    inner = SimpleNamespace(embed_tokens=SimpleNamespace(weight=param))
    target = SimpleNamespace(embed_tokens=None)
    weight = torch.ones(2, 3)
    source_calls = []

    class Loader:
        def __init__(self, config):
            pass

        @staticmethod
        def Source(model, revision):
            source_calls.append((model, revision))
            return None

        def _get_weights_iterator(self, source):
            yield "model.language_model.embed_tokens.weight", weight

    monkeypatch.setattr(utils, "DefaultModelLoader", Loader)
    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=8))
    config = SimpleNamespace(
        load_config=None,
        model_config=SimpleNamespace(model="pinned-target", revision="fixed-revision"),
    )
    utils._load_pp_target_embedding(draft, inner, target, config)
    assert loaded == ([] if has_own else [weight])
    assert source_calls == ([] if has_own else [("pinned-target", "fixed-revision")])
    assert draft.has_own_embed_tokens
