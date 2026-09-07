# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False
    # The packed FlashInfer ABI reserves a 64-element BF16 tail.
    _packed_rope_head_dim = 64

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        if self.kv_lora_rank != 512 or self.qk_rope_head_dim not in (0, 64):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires kv_lora_rank=512 and "
                "qk_rope_head_dim in (0, 64); got "
                f"kv_lora_rank={self.kv_lora_rank} and "
                f"qk_rope_head_dim={self.qk_rope_head_dim}."
            )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            model_type = getattr(hf_text_config, "model_type", None)
            self.sparse_mla_top_k = int(getattr(hf_text_config, "index_topk", 2048))
            self.index_kpool = int(getattr(hf_text_config, "index_kpool", 1) or 1)
        else:
            self.sparse_mla_top_k = 2048
            self.index_kpool = 1
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Skip-topk layers are built with indexer=None and get the shared
        # buffer via mla_args instead (cf. FLASHMLA_SPARSE).
        self._indexer = indexer
        self._topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None
        self._topk_columns = torch.arange(
            self.sparse_mla_top_k,
            dtype=torch.int64,
            device=self.topk_indices_buffer.device,
        ).unsqueeze(0)

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

    @property
    def topk_indices_buffer(self) -> torch.Tensor | None:
        if self._indexer is not None:
            return self._indexer.topk_indices_buffer
        return self._topk_indices_buffer

    @topk_indices_buffer.setter
    def topk_indices_buffer(self, buffer: torch.Tensor | None) -> None:
        self._topk_indices_buffer = buffer
        if self._indexer is not None:
            self._indexer.topk_indices_buffer = buffer

    def _fit_topk_indices(self, topk_indices: torch.Tensor) -> torch.Tensor:
        topk = self.sparse_mla_top_k
        if topk_indices.shape[1] == topk:
            return topk_indices

        tail_width = min(self.index_kpool - 1, topk_indices.shape[1] - topk)
        if tail_width <= 0:
            return topk_indices[:, :topk]

        history = topk_indices[:, :topk]
        tail = topk_indices[:, topk : topk + tail_width]
        tail_counts = (tail >= 0).sum(dim=1, keepdim=True)
        history_limits = topk - tail_counts
        tail_offsets = (self._topk_columns - history_limits).clamp(
            min=0, max=tail_width - 1
        )
        tail_values = torch.gather(tail, 1, tail_offsets)
        return torch.where(self._topk_columns < history_limits, history, tail_values)

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if self.qk_rope_head_dim == 0:
            k_pe = F.pad(k_pe, (0, self._packed_rope_head_dim))
        super().do_kv_cache_update(
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if self.qk_rope_head_dim == 0:
            q = F.pad(q, (0, self._packed_rope_head_dim))

        num_actual_toks = q.shape[0]
        actual_num_heads = q.shape[1]
        kernel_num_heads = max(8, 1 << (actual_num_heads - 1).bit_length())
        if kernel_num_heads > 128:
            raise ValueError(
                "FLASHINFER_MLA_SPARSE_SM120 supports at most 128 attention "
                f"heads per worker; got {actual_num_heads}."
            )
        if kernel_num_heads != actual_num_heads:
            q_padded = q.new_zeros((num_actual_toks, kernel_num_heads, q.shape[-1]))
            q_padded[:, :actual_num_heads].copy_(q)
            q = q_padded

        assert self.topk_indices_buffer is not None
        topk_indices = self._fit_topk_indices(
            self.topk_indices_buffer[:num_actual_toks]
        )

        topk_indices_physical, topk_lengths = cast(
            tuple[torch.Tensor, torch.Tensor],
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=self.sparse_mla_top_k,
                return_valid_counts=True,
            ),
        )

        output = q.new_empty(
            (num_actual_toks, kernel_num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=(
                self._packed_rope_head_dim
                if self.qk_rope_head_dim == 0
                else self.qk_rope_head_dim
            ),
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=topk_lengths,
            max_seq_len=self.sparse_mla_top_k,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=self.sparse_mla_top_k,
            kv_scale_format=self.kv_scale_format,
        )
        return out.squeeze(1)[:, :actual_num_heads], None
