# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.quantization.utils import flashinfer_fp4_moe
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_nvfp4_moe_layer_for_fi_or_cutlass,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    align_fp4_moe_weights_for_fi,
    align_trtllm_fp4_moe_hidden_dim_for_fi,
)


def test_shared_nvfp4_input_scales_have_writable_storage(monkeypatch):
    monkeypatch.setattr(flashinfer_fp4_moe, "swizzle_blockscale", lambda x: x)

    num_experts = 3
    layer = SimpleNamespace(
        activation=SimpleNamespace(is_gated=False),
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=False)
        ),
    )
    hidden_dim = 32
    w13 = torch.zeros((num_experts, 2, hidden_dim // 2), dtype=torch.uint8)
    w2 = torch.zeros((num_experts, hidden_dim, 1), dtype=torch.uint8)
    w13_scale = torch.zeros(
        (num_experts, 2, hidden_dim // 16), dtype=torch.float8_e4m3fn
    )
    w2_scale = torch.zeros((num_experts, hidden_dim, 1), dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones(num_experts)

    outputs = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
        backend=NvFp4MoeBackend.FLASHINFER_CUTLASS,
        layer=layer,
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=weight_scale,
        a13_scale=torch.tensor([1.0, 2.0, 3.0]),
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=weight_scale,
        a2_scale=torch.tensor([4.0, 5.0, 6.0]),
        is_act_and_mul=False,
    )
    a13_scale, a2_scale = outputs[3], outputs[7]

    torch.testing.assert_close(a13_scale, torch.full((num_experts,), 3.0))
    torch.testing.assert_close(a2_scale, torch.full((num_experts,), 6.0))
    distinct_values = torch.arange(num_experts, dtype=torch.float32)
    a13_scale.copy_(distinct_values)
    a2_scale.copy_(distinct_values)
    torch.testing.assert_close(a13_scale, distinct_values)
    torch.testing.assert_close(a2_scale, distinct_values)


def test_align_trtllm_fp4_moe_hidden_dim_noop():
    w13 = torch.arange(2 * 8 * 256, dtype=torch.uint8).reshape(2, 8, 256)
    w13_scale = torch.arange(2 * 8 * 32, dtype=torch.uint8).reshape(2, 8, 32)
    w2 = torch.arange(2 * 512 * 4, dtype=torch.uint8).reshape(2, 512, 4)
    w2_scale = torch.arange(2 * 512 * 1, dtype=torch.uint8).reshape(2, 512, 1)

    out_w13, out_w13_scale, out_w2, out_w2_scale, padded_hidden = (
        align_trtllm_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
    )

    assert padded_hidden == 512
    assert out_w13 is w13
    assert out_w13_scale is w13_scale
    assert out_w2 is w2
    assert out_w2_scale is w2_scale


def test_align_trtllm_fp4_moe_hidden_dim_pads_to_256_multiple():
    hidden_dim = 2688
    padded_hidden_dim = 2816

    w13 = torch.arange(2 * 12 * (hidden_dim // 2), dtype=torch.uint8).reshape(
        2, 12, hidden_dim // 2
    )
    w13_scale = torch.arange(2 * 12 * (hidden_dim // 16), dtype=torch.uint8).reshape(
        2, 12, hidden_dim // 16
    )

    w2 = torch.arange(2 * hidden_dim * 6, dtype=torch.uint8).reshape(2, hidden_dim, 6)
    w2_scale = torch.arange(2 * hidden_dim * 2, dtype=torch.uint8).reshape(
        2, hidden_dim, 2
    )

    out_w13, out_w13_scale, out_w2, out_w2_scale, out_hidden_dim = (
        align_trtllm_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
    )

    assert out_hidden_dim == padded_hidden_dim
    assert out_w13.shape == (2, 12, padded_hidden_dim // 2)
    assert out_w13_scale.shape == (2, 12, padded_hidden_dim // 16)
    assert out_w2.shape == (2, padded_hidden_dim, 6)
    assert out_w2_scale.shape == (2, padded_hidden_dim, 2)

    torch.testing.assert_close(out_w13[:, :, : hidden_dim // 2], w13)
    torch.testing.assert_close(out_w13_scale[:, :, : hidden_dim // 16], w13_scale)
    torch.testing.assert_close(out_w2[:, :hidden_dim, :], w2)
    torch.testing.assert_close(out_w2_scale[:, :hidden_dim, :], w2_scale)

    assert torch.count_nonzero(out_w13[:, :, hidden_dim // 2 :]) == 0
    assert torch.count_nonzero(out_w13_scale[:, :, hidden_dim // 16 :]) == 0
    assert torch.count_nonzero(out_w2[:, hidden_dim:, :]) == 0
    assert torch.count_nonzero(out_w2_scale[:, hidden_dim:, :]) == 0


def test_align_trtllm_fp4_moe_intermediate_pads_gate_and_up_separately():
    # Qwen3.8-Flash-Next has intermediate_size=640. At TP4 each rank owns
    # 160 values, which TRTLLM pads to the next 64-value boundary (192).
    intermediate = 160
    padded_intermediate = 192
    hidden_dim = 32

    gate = torch.ones((1, intermediate, hidden_dim // 2), dtype=torch.uint8)
    up = torch.full_like(gate, 2)
    w13 = torch.cat((gate, up), dim=1)
    gate_scale = torch.full((1, intermediate, hidden_dim // 16), 3, dtype=torch.uint8)
    up_scale = torch.full_like(gate_scale, 4)
    w13_scale = torch.cat((gate_scale, up_scale), dim=1)
    w2 = torch.full((1, hidden_dim, intermediate // 2), 5, dtype=torch.uint8)
    w2_scale = torch.full((1, hidden_dim, intermediate // 16), 6, dtype=torch.uint8)

    out_w13, out_w13_scale, out_w2, out_w2_scale, out_intermediate = (
        align_fp4_moe_weights_for_fi(
            w13,
            w13_scale,
            w2,
            w2_scale,
            is_act_and_mul=True,
            min_alignment=64,
        )
    )

    assert out_intermediate == padded_intermediate
    torch.testing.assert_close(out_w13[:, :intermediate], gate)
    torch.testing.assert_close(
        out_w13[:, padded_intermediate : padded_intermediate + intermediate], up
    )
    torch.testing.assert_close(out_w13_scale[:, :intermediate], gate_scale)
    torch.testing.assert_close(
        out_w13_scale[:, padded_intermediate : padded_intermediate + intermediate],
        up_scale,
    )

    assert torch.count_nonzero(out_w13[:, intermediate:padded_intermediate]) == 0
    assert torch.count_nonzero(out_w13[:, padded_intermediate + intermediate :]) == 0
    assert torch.count_nonzero(out_w13_scale[:, intermediate:padded_intermediate]) == 0
    assert (
        torch.count_nonzero(out_w13_scale[:, padded_intermediate + intermediate :]) == 0
    )
    torch.testing.assert_close(out_w2[:, :, : intermediate // 2], w2)
    torch.testing.assert_close(out_w2_scale[:, :, : intermediate // 16], w2_scale)
    assert torch.count_nonzero(out_w2[:, :, intermediate // 2 :]) == 0
    assert torch.count_nonzero(out_w2_scale[:, :, intermediate // 16 :]) == 0


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    "backend",
    [NvFp4MoeBackend.FLASHINFER_CUTLASS, NvFp4MoeBackend.VLLM_CUTLASS],
)
def test_cutlass_padding_preserves_each_projection_and_scale(
    monkeypatch, tp, gated, backend
):
    """Qwen TP shards retain both projections when scale rows need padding."""

    def swizzle_pad_rows(scale):
        assert scale.size(1) % 128 == 0
        return scale

    monkeypatch.setattr(flashinfer_fp4_moe, "swizzle_blockscale", swizzle_pad_rows)
    intermediate = 640 // tp
    alignment = 64 if gated else 128
    padded = (intermediate + alignment - 1) // alignment * alignment
    hidden_dim = 128
    multiplier = 2 if gated else 1
    w13 = torch.ones((1, multiplier * intermediate, hidden_dim // 2), dtype=torch.uint8)
    w13_scale = torch.full(
        (1, multiplier * intermediate, hidden_dim // 16), 3, dtype=torch.uint8
    )
    if gated:
        w13[:, intermediate:] = 2
        w13_scale[:, intermediate:] = 4
    w2 = torch.full((1, hidden_dim, intermediate // 2), 5, dtype=torch.uint8)
    w2_scale = torch.full((1, hidden_dim, intermediate // 16), 6, dtype=torch.uint8)
    layer = SimpleNamespace(
        activation=SimpleNamespace(is_gated=gated),
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=False),
            intermediate_size_per_partition=intermediate,
        ),
    )
    weight_scale = torch.ones(1)
    original13, original_scale13 = w13.clone(), w13_scale.clone()
    outputs = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
        backend,
        layer,
        w13,
        w13_scale,
        weight_scale,
        torch.ones(1, 2),
        w2,
        w2_scale,
        weight_scale,
        weight_scale,
        gated,
    )
    out13, scale13, _, _, out2, scale2, _, _ = outputs
    assert layer.moe_config.intermediate_size_per_partition == padded
    assert out13.shape == (1, multiplier * padded, hidden_dim // 2)
    reordered = gated and backend == NvFp4MoeBackend.FLASHINFER_CUTLASS
    for projection in range(multiplier):
        source = (1 - projection) if reordered else projection
        destination = slice(projection * padded, projection * padded + intermediate)
        original = slice(source * intermediate, (source + 1) * intermediate)
        torch.testing.assert_close(out13[:, destination], original13[:, original])
        torch.testing.assert_close(
            scale13[:, destination], original_scale13[:, original]
        )
        tail = slice(projection * padded + intermediate, (projection + 1) * padded)
        assert torch.count_nonzero(out13[:, tail]) == 0
        assert torch.count_nonzero(scale13[:, tail]) == 0
    torch.testing.assert_close(out2[:, :, : intermediate // 2], w2)
    torch.testing.assert_close(scale2[:, :, : intermediate // 16], w2_scale)
    assert torch.count_nonzero(out2[:, :, intermediate // 2 :]) == 0
    assert torch.count_nonzero(scale2[:, :, intermediate // 16 :]) == 0
    torch.testing.assert_close(outputs[2], weight_scale)
    torch.testing.assert_close(outputs[6], weight_scale)
    if padded == intermediate:
        assert out2 is w2
        assert scale2 is w2_scale
