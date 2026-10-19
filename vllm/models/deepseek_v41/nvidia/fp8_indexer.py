# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-only FP8 index scoring on consumer Blackwell.

The prefill, short-context and non-flat decode paths retain the dense indexer.
This path uses the same candidate set, FP8 inputs and FP32 scores as that path.
"""

import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import current_workspace_manager


@triton.jit
def _sort_candidates(source, output, SOURCE_STRIDE: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, K)
    value = tl.load(source + row * SOURCE_STRIDE + col)
    value = tl.sort(tl.where(value >= 0, value, 0x7FFFFFFF))
    tl.store(output + row * K + col, value)


@triton.jit
def _score_candidates(
    q,
    cache,
    weights,
    block_table,
    seq_lens,
    candidates,
    logits,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    W_ROW: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    CACHE_PAGE: tl.constexpr,
    PAGE: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    HEADS: tl.constexpr,
    TILE: tl.constexpr,
    WIDTH: tl.constexpr,
    DENSE_SHORT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    head = tl.arange(0, HEADS)
    dim = tl.arange(0, 128)
    candidate = tl.load(
        candidates + row * K + cols // BLOCK, cols < K * BLOCK, other=0x7FFFFFFF
    ).to(tl.int64)
    token = candidate * BLOCK + cols % BLOCK
    length = tl.load(seq_lens + row)
    valid = (candidate != 0x7FFFFFFF) & (token < length)
    if DENSE_SHORT and length < 32768:
        # Preserve dense column numbering for short rows, even when a
        # FULL graph was captured on a long-context dummy batch.
        token = cols.to(tl.int64)
        lo = tl.full((TILE,), 0, tl.int32)
        hi = tl.full((TILE,), K, tl.int32)
        for _ in range(12):
            mid = (lo + hi) // 2
            block = tl.load(candidates + row * K + mid, mid < K, other=0x7FFFFFFF)
            before = block < cols // BLOCK
            lo = tl.where(before, mid + 1, lo)
            hi = tl.where(before, hi, mid)
        block = tl.load(candidates + row * K + lo, lo < K, other=0x7FFFFFFF)
        valid = (token < length) & (block == cols // BLOCK)
    page = tl.load(block_table + row * TABLE_ROW + token // PAGE, valid, other=0).to(
        tl.int64
    )
    offset = token % PAGE
    key = tl.load(
        cache + page[None, :] * CACHE_PAGE + offset[None, :] * 128 + dim[:, None],
        valid[None, :],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    scale_ptr = cache + page * CACHE_PAGE + PAGE * 128 + offset * 4
    bits = tl.load(scale_ptr, valid, other=0).to(tl.uint32)
    bits |= tl.load(scale_ptr + 1, valid, other=0).to(tl.uint32) << 8
    bits |= tl.load(scale_ptr + 2, valid, other=0).to(tl.uint32) << 16
    bits |= tl.load(scale_ptr + 3, valid, other=0).to(tl.uint32) << 24
    scale = bits.to(tl.float32, bitcast=True)
    query = tl.load(q + row * Q_ROW + head[:, None] * Q_HEAD + dim[None, :])
    score = tl.dot(query, key, out_dtype=tl.float32)
    weight = tl.load(weights + row * W_ROW + head)
    score = (
        tl.sum(
            tl.maximum(score, 0.0, propagate_nan=tl.PropagateNan.ALL) * weight[:, None],
            0,
        )
        * scale
    )
    tl.store(logits + row * WIDTH + cols, tl.where(valid, score, -float("inf")))


@triton.jit
def _remap_candidates(
    indices,
    candidates,
    logits,
    output,
    seq_lens,
    OUTPUT_ROW: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    DENSE_SHORT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, triton.next_power_of_2(TOPK))
    selected = tl.load(indices + row * TOPK + col, col < TOPK, other=-1)
    score = tl.load(
        logits + row * WIDTH + selected,
        (col < TOPK) & (selected >= 0) & (selected < WIDTH),
        other=-float("inf"),
    )
    block = tl.load(
        candidates + row * K + selected // BLOCK,
        (col < TOPK) & (selected >= 0) & (selected < K * BLOCK),
        other=0x7FFFFFFF,
    )
    valid = (selected >= 0) & (score != -float("inf")) & (block != 0x7FFFFFFF)
    token = block.to(tl.int64) * BLOCK + selected % BLOCK
    if DENSE_SHORT and tl.load(seq_lens + row) < 32768:
        token = selected.to(tl.int64)
        valid = (selected >= 0) & (score != -float("inf"))
    tl.store(output + row * OUTPUT_ROW + col, tl.where(valid, token, -1), col < TOPK)


def candidate_scores(
    q,
    cache,
    weights,
    block_table,
    seq_lens,
    candidates,
    sorted_candidates,
    logits,
    block_size,
    dense_short_context=False,
):
    """Write compact FP32 scores; inputs use the native segregated FP8 pages."""
    rows, heads, dim = q.shape
    if dim != 128 or heads != 32 or candidates.shape[1] != 2048:
        raise ValueError("SM120 candidate scoring requires H=32, D=128, K=2048")
    if q.dtype != torch.float8_e4m3fn or cache.dtype != torch.uint8:
        raise ValueError("SM120 candidate scoring requires FP8 Q and packed FP8 K")
    if block_size != 8:
        raise ValueError("SM120 candidate scoring requires eight-token candidates")
    if logits.shape[1] != (32768 if dense_short_context else 16384):
        raise ValueError("Candidate score workspace width does not match its layout")
    if not rows:
        return
    _sort_candidates[(rows,)](candidates, sorted_candidates, candidates.stride(0), 2048)
    _score_candidates[(rows, triton.cdiv(logits.shape[1], 64))](
        q,
        cache,
        weights,
        block_table,
        seq_lens,
        sorted_candidates,
        logits,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        block_table.stride(0),
        cache.stride(0),
        cache.shape[1],
        2048,
        8,
        32,
        64,
        logits.shape[1],
        dense_short_context,
        num_warps=4,
    )


class FP8CandidateIndexer(SparseAttnIndexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = get_current_vllm_config()
        self.max_rows = cfg.scheduler_config.max_num_batched_tokens
        self._warmed = False
        if self.dcp_world_size != 1 or self.use_pcp or self.use_fp4_cache:
            raise ValueError("SM120 candidate scoring requires FP8 without CP")

    def _workspace(self, rows):
        return current_workspace_manager().get_simultaneous(
            ((rows, 32768), torch.float32),
            ((rows, 2048), torch.int32),
            ((rows, self.topk_tokens), torch.int32),
            ((rows,), torch.int32),
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

    def forward_cuda(self, hidden_states, q_quant, k, weights):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            self._workspace(self.max_rows)
            return super().forward_cuda(hidden_states, q_quant, k, weights)
        metadata = metadata[self.k_cache.prefix]
        decode = metadata.decode
        rows = metadata.num_decode_tokens
        if (
            metadata.num_prefills
            or not rows
            or (
                metadata.max_seq_len < 32768
                and self._warmed
                and not torch.cuda.is_current_stream_capturing()
            )
            or decode is None
            or decode.indices is not None
            or decode.seq_lens.numel() != rows
            or decode.block_table.shape[0] != rows
            or isinstance(q_quant, tuple)
        ):
            return super().forward_cuda(hidden_states, q_quant, k, weights)
        logits, candidates, indices, ends, scratch = self._workspace(rows)
        candidate_scores(
            q_quant[:rows],
            self.k_cache.kv_cache,
            weights[:rows],
            decode.block_table,
            decode.seq_lens,
            self.candidate_blocks[:rows],
            candidates,
            logits,
            self.candidate_block_size,
            dense_short_context=True,
        )
        self._warmed = True
        ends.fill_(32768)
        torch.ops._C.persistent_topk(
            logits, ends, indices, scratch, self.topk_tokens, 32768
        )
        _remap_candidates[(rows,)](
            indices,
            candidates,
            logits,
            self.topk_indices_buffer,
            decode.seq_lens,
            self.topk_indices_buffer.stride(0),
            2048,
            8,
            self.topk_tokens,
            32768,
            True,
        )
        return self.topk_indices_buffer
