# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-only incremental transfer of shared DeepSeek V4.1 cache records."""

import torch
from torch import nn

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v41.common.pp import shared_cache_dependency
from vllm.triton_utils import tl, triton


class SharedKVCache(nn.Module, AttentionLayerBase):
    """A weight-free, stage-local mirror of an upstream compressed cache."""

    def __init__(self, prefix, spec, backend, compilation_config):
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.backend = backend
        self.kv_cache = torch.tensor([])
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache):
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        return self.backend

    def forward(self):
        raise RuntimeError("A shared cache mirror does not compute attention")


@triton.jit
def _copy_packed_rows(
    cache,
    slots,
    positions,
    rows,
    num_slots,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    DATA: tl.constexpr,
    SCALES: tl.constexpr,
    RATIO: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    INSERT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row, row < num_slots, other=-1).to(tl.int64)
    pos = tl.load(positions + row)
    valid = (slot >= 0) & (pos >= 0) & ((pos + 1) % RATIO == 0)
    cols = tl.arange(0, TILE)
    offset = tl.where(
        cols < DATA,
        slot % PAGE * DATA + cols,
        PAGE * DATA + slot % PAGE * SCALES + cols - DATA,
    )
    ptr = cache + slot // PAGE * STRIDE + offset
    packed = rows + row * ROW_STRIDE + cols
    if INSERT:
        valid = valid & (tl.load(rows + row * ROW_STRIDE + DATA + SCALES) != 0)
        value = tl.load(packed, (cols < DATA + SCALES) & valid, other=0)
        tl.store(ptr, value, (cols < DATA + SCALES) & valid)
    else:
        value = tl.load(ptr, (cols < DATA + SCALES) & valid, other=0)
        tl.store(packed, value, cols < DATA + SCALES)
        tl.store(rows + row * ROW_STRIDE + DATA + SCALES, valid.to(tl.uint8))


def copy_packed_rows(cache, slots, positions, rows, data_bytes, ratio, *, insert):
    """Copy data and scale bytes without reading invalid or unclosed groups."""
    if rows.shape[0] == 0:
        return
    if cache.dtype != torch.uint8 or rows.dtype != torch.uint8:
        raise ValueError("PP cache transport requires packed uint8 cache records")
    scale_bytes = rows.shape[1] - data_bytes - 1
    _copy_packed_rows[(rows.shape[0],)](
        cache,
        slots,
        positions,
        rows,
        slots.numel(),
        cache.shape[1],
        cache.stride(0),
        data_bytes,
        scale_bytes,
        ratio,
        rows.stride(0),
        insert,
        triton.next_power_of_2(rows.shape[1]),
    )


class PPSharedState(nn.Module):
    KV = "pp_compressed_kv"
    INDEX = "pp_indexer_k"
    TOPK = "pp_topk_indices"
    CANDIDATES = "pp_candidate_blocks"

    def __init__(self, vllm_config, config, prefix, start, end):
        super().__init__()
        self.incoming = shared_cache_dependency(config, start)
        self.outgoing = shared_cache_dependency(config, end)
        self.context = vllm_config.compilation_config.static_forward_context
        self.prefix = prefix
        self.index_bytes = config.index_head_dim + config.index_head_dim // 128 * 4
        from vllm.models.deepseek_v41.attention import _use_v41_mxfp8_kv_record

        self.kv_bytes = 528 if _use_v41_mxfp8_kv_record() else 584
        self.kv_data_bytes = 512 if self.kv_bytes == 528 else 576
        if self.incoming or self.outgoing:
            if vllm_config.attention_config.indexer_kv_dtype == "mxfp4":
                raise ValueError(
                    "Cross-stage V4.1 cache sharing requires FP8 index keys"
                )
            if vllm_config.cache_config.cache_dtype != "fp8_ds_mla":
                raise ValueError(
                    "Cross-stage V4.1 cache sharing requires fp8_ds_mla KV"
                )
            if not vllm_config.use_v2_model_runner:
                raise ValueError(
                    "Cross-stage V4.1 cache sharing requires model runner V2"
                )
            if vllm_config.parallel_config.use_ubatching:
                raise ValueError("Cross-stage V4.1 cache sharing does not support DBO")
            parallel = vllm_config.parallel_config
            if (
                parallel.decode_context_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
            ):
                raise ValueError("Cross-stage V4.1 cache sharing does not support CP")
        self._out = self.empty_tensors(
            self.outgoing,
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            getattr(config, "candidate_topk_blocks", 0),
        )

    def empty_tensors(self, dependency, count, topk, candidates, device=None):
        if dependency is None:
            return {}
        tensors = {
            self.KV: torch.zeros(
                (count, self.kv_bytes + 1), dtype=torch.uint8, device=device
            ),
            self.INDEX: torch.zeros(
                (count, self.index_bytes + 1), dtype=torch.uint8, device=device
            ),
            self.TOPK: torch.full((count, topk), -1, dtype=torch.int32, device=device),
        }
        if dependency.candidates:
            tensors[self.CANDIDATES] = torch.full(
                (count, candidates),
                -1,
                dtype=torch.int32,
                device=device,
            )
        return tensors

    def _copy(self, dependency, tensors, positions, *, insert):
        if not is_forward_context_available():
            return
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return
        source = f"{self.prefix}.layers.{dependency.source}.attn"
        for key, name, data in (
            (self.KV, source, self.kv_data_bytes),
            (self.INDEX, source + ".indexer.k_cache", self.index_bytes - 4),
        ):
            cache = self.context[name].kv_cache
            if cache.numel() == 0:
                if not insert:
                    tensors[key].zero_()
                continue
            if name not in metadata:
                raise RuntimeError(f"Missing PP shared-cache metadata for {name}")
            copy_packed_rows(
                cache,
                metadata[name].slot_mapping,
                positions,
                tensors[key],
                data,
                dependency.compress_ratio,
                insert=insert,
            )
            probe = getattr(self, "diagnostic_probe", None)
            if insert and probe is not None and probe.active:
                from .diagnostics import check_received_rows

                replay = torch.empty_like(tensors[key])
                copy_packed_rows(
                    cache,
                    metadata[name].slot_mapping,
                    positions,
                    replay,
                    data,
                    dependency.compress_ratio,
                    insert=False,
                )
                compared = check_received_rows(
                    tensors[key],
                    replay,
                    metadata[name].slot_mapping,
                    positions,
                    dependency.compress_ratio,
                )
                probe.emit("cache_insert", name=name, byte_identical_rows=compared)

    def receive(self, intermediate, positions, topk, candidates):
        if self.incoming is None:
            return
        self._copy(self.incoming, intermediate.tensors, positions, insert=True)
        count = positions.shape[0]
        topk[:count].copy_(intermediate[self.TOPK])
        if self.incoming.candidates:
            candidates[:count].copy_(intermediate[self.CANDIDATES])
        probe = getattr(self, "diagnostic_probe", None)
        if probe is not None and probe.active:
            if not torch.equal(topk[:count], intermediate[self.TOPK]):
                raise RuntimeError("DeepSeek received top-k selection differs")
            if self.incoming.candidates and not torch.equal(
                candidates[:count], intermediate[self.CANDIDATES]
            ):
                raise RuntimeError("DeepSeek received candidate selection differs")

    def send(self, positions, topk, candidates):
        if self.outgoing is None:
            return {}
        count = positions.shape[0]
        tensors = {key: value[:count] for key, value in self._out.items()}
        self._copy(self.outgoing, tensors, positions, insert=False)
        tensors[self.TOPK].copy_(topk[:count])
        if self.outgoing.candidates:
            tensors[self.CANDIDATES].copy_(candidates[:count])
        return tensors
