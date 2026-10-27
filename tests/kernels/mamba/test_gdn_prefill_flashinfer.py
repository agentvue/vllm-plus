# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        reason="FlashInfer GDN prefill is not supported on ROCm.",
        allow_module_level=True,
    )

import flashinfer.gdn_prefill  # noqa: E402

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    fi_chunk_gated_delta_rule,
)  # noqa: E402


@pytest.mark.parametrize("fail", [False, True])
def test_gdn_warmup_profiles_full_scheduler_batch(monkeypatch, fail):
    """A short MTP dummy batch must not understate prefill memory."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    calls = []

    def post_conv_prep(**kwargs):
        T = kwargs["conv_output"].shape[0]
        q = torch.zeros(T, 1, 2)
        return q, q, q, torch.zeros(T, 1), torch.zeros(T, 1)

    def prefill(**kwargs):
        calls.append(kwargs)
        if fail:
            raise RuntimeError("warmup failed")

    monkeypatch.setattr(gdn, "fused_post_conv_prep", post_conv_prep)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    owner = SimpleNamespace(
        _prefill_kernels_warmed_up=False,
        _prefill_profile_num_tokens=257,
        _prefill_profile_num_reqs=4,
        num_k_heads=1,
        num_v_heads=1,
        tp_size=1,
        head_v_dim=2,
        head_k_dim=2,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        get_state_dtype=lambda: (torch.float32, torch.float32),
        gdn_prefill_backend="flashinfer",
        chunk_gated_delta_rule=prefill,
        prefix="layer",
    )
    warmup = lambda: QwenGatedDeltaNetAttention._warmup_prefill_kernels(
        owner, torch.zeros(233, 6), 0
    )
    if fail:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup()
        assert not owner._prefill_kernels_warmed_up
    else:
        warmup()
        warmup()
        assert owner._prefill_kernels_warmed_up
        assert len(calls) == 1
    assert calls[0]["q"].shape[1] == 257
    assert calls[0]["cu_seqlens"].tolist() == [0, 64, 128, 192, 257]
    assert calls[0]["initial_state"].shape == (4, 1, 2, 2)


def test_flashinfer_gdn_prefill_uses_int64_cu_seqlens(monkeypatch):
    captured_cu_seqlens = None

    def fake_chunk_gated_delta_rule(**kwargs):
        nonlocal captured_cu_seqlens
        captured_cu_seqlens = kwargs["cu_seqlens"]
        return kwargs["q"]

    monkeypatch.setattr(
        flashinfer.gdn_prefill,
        "chunk_gated_delta_rule",
        fake_chunk_gated_delta_rule,
    )
    q = torch.zeros(1, 2, 1, 2)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    output, final_state = fi_chunk_gated_delta_rule(
        q=q,
        k=q,
        v=q,
        g=torch.zeros(1, 2, 1),
        beta=torch.zeros(1, 2, 1),
        initial_state=torch.zeros(1, 1, 2, 2),
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    assert captured_cu_seqlens is not None
    assert captured_cu_seqlens.dtype == torch.int64
    assert output.shape == q.shape
    assert final_state is None
