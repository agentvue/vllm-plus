# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed cache transport preserves bytes across distinct physical page layouts."""

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.pp import copy_packed_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("data,scales", [(576, 8), (512, 16), (128, 4)])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("capture", [False, True])
def test_cache_updates_preserve_values_scales_and_skipped_slots(
    data, scales, ratio, capture
):
    src_page, dst_page = 128, 64
    width = data + scales
    src_stride, dst_stride = src_page * width + 256, dst_page * width + 512
    src_raw = (torch.arange(3 * src_stride, device="cuda") % 251).to(torch.uint8)
    dst_raw = torch.full((5 * dst_stride,), 253, dtype=torch.uint8, device="cuda")
    source = src_raw.as_strided((3, src_page, width), (src_stride, width, 1))
    target = dst_raw.as_strided((5, dst_page, width), (dst_stride, width, 1))
    slots = torch.tensor([0, 127, -1, 128, 129, 256, 257, 258], device="cuda")
    target_slots = torch.tensor([63, 64, 65, 128, -1, 191, 192, 193], device="cuda")
    positions = torch.tensor([1, 2, 3, 4, 5, 127, 128, 129], device="cuda")
    packed = torch.empty((8, width + 1), dtype=torch.uint8, device="cuda")

    def update():
        copy_packed_rows(source, slots, positions, packed, data, ratio, insert=False)
        copy_packed_rows(
            target, target_slots, positions, packed, data, ratio, insert=True
        )

    update()
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            update()
        for _ in range(3):
            graph.replay()
    # Reused request slots and new compression boundaries must use current data.
    dst_raw.fill_(253)
    slots.copy_(slots.roll(1))
    target_slots.copy_(target_slots.roll(1))
    positions.add_(1)
    if capture:
        graph.replay()
    else:
        update()
    torch.cuda.synchronize()
    expected = torch.full_like(dst_raw, 253).cpu()
    src = src_raw.cpu()
    for src_slot, dst_slot, pos in zip(
        slots.tolist(), target_slots.tolist(), positions.tolist()
    ):
        if src_slot < 0 or dst_slot < 0 or (pos + 1) % ratio:
            continue
        for count, src_offset, dst_offset in (
            (data, (src_slot % src_page) * data, (dst_slot % dst_page) * data),
            (
                scales,
                src_page * data + (src_slot % src_page) * scales,
                dst_page * data + (dst_slot % dst_page) * scales,
            ),
        ):
            src_offset += src_slot // src_page * src_stride
            dst_offset += dst_slot // dst_page * dst_stride
            expected[dst_offset : dst_offset + count] = src[
                src_offset : src_offset + count
            ]
    assert torch.equal(dst_raw.cpu(), expected)


@pytest.mark.parametrize("num_tokens", [1, 32, 65])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_sm120_masked_kv_cannot_poison_attention_from_null_page(
    num_tokens, capture, compressed
):
    """Exercise real FlashInfer decode/prefill with two NaNs in reserved block 0."""
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM120 sparse MLA")
    from vllm.models.deepseek_v41.nvidia.diagnostics import sparse_attention_reference
    from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferSM120Attention,
    )
    from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4

    page, stride = 64, 92160
    raw = torch.full((2, stride), 255, device="cuda", dtype=torch.uint8)
    cache = raw.as_strided((2, page, 584), (stride, 584, 1))
    data = raw[1, : page * 576].view(page, 576)
    data[:, :448].copy_(
        torch.ones(page, 448, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    )
    data[:, 448:].view(torch.bfloat16).fill_(1)
    raw[1, page * 576 : page * 584].fill_(127)
    live = cache[1:].clone()
    extra_raw = raw.clone() if compressed else None
    extra_cache = (
        extra_raw.as_strided(cache.shape, cache.stride()) if compressed else None
    )
    q = torch.full((num_tokens, 8, 512), 1 / 64, device="cuda", dtype=torch.bfloat16)
    indices = torch.full((num_tokens, 128), -1, device="cuda", dtype=torch.int32)
    indices[:, :2] = torch.tensor([64, 65], device="cuda", dtype=torch.int32)
    extra_indices = None
    if compressed:
        extra_indices = torch.full(
            (num_tokens, 512), -1, device="cuda", dtype=torch.int32
        )
        extra_indices[:, :2] = indices[:, :2]
    lengths = torch.full((num_tokens,), 2, device="cuda", dtype=torch.int32)
    sinks = torch.full((8,), -2.0, device="cuda")
    output = torch.empty_like(q)
    workspace = torch.zeros(128 * 1024**2, dtype=torch.uint8, device="cuda")
    scale = 512**-0.5

    def run():
        cache[0].zero_()
        raw[0, 448:452].view(torch.bfloat16).fill_(float("nan"))
        prepared = DeepseekV4FlashInferSM120Attention._as_sparse_cache(cache)
        prepared_extra = None
        if compressed:
            extra_cache[0].zero_()
            extra_raw[0, 448:452].view(torch.bfloat16).fill_(float("nan"))
            prepared_extra = DeepseekV4FlashInferSM120Attention._as_sparse_cache(
                extra_cache
            )
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=prepared,
            workspace_buffer=workspace,
            sparse_indices=indices,
            out=output,
            bmm1_scale=scale,
            sinks=sinks,
            kv_layout="NHD",
            swa_topk_lens=lengths,
            compressed_kv_cache=prepared_extra,
            extra_sparse_indices=extra_indices,
            extra_sparse_topk_lens=lengths if compressed else None,
        )

    run()
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(3):
            graph.replay()
    reference_indices = (
        torch.cat((indices[:, :2], extra_indices[:, :2]), -1) if compressed else indices
    )
    reference = sparse_attention_reference(
        q, cache, reference_indices, lengths * (2 if compressed else 1), scale, sinks
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:2].float(), reference, rtol=0.03, atol=0.02)
    assert torch.equal(cache[1:], live)
    if compressed:
        assert torch.equal(extra_cache[1:], live)
    assert (raw[0, page * 584 :] == 255).all()
