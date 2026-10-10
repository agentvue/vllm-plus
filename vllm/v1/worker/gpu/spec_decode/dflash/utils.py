# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.model_executor.models.utils import PPMissingLayer, get_draft_quant_config
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config


def _load_pp_target_embedding(
    draft_model: nn.Module,
    draft_inner: nn.Module,
    target_inner: nn.Module,
    vllm_config: VllmConfig,
) -> None:
    draft = cast(Any, draft_model)
    target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
        target_inner, "embedding", None
    )
    if (
        get_pp_group().world_size <= 1
        or getattr(draft_model, "has_own_embed_tokens", False)
        or (target_embed is not None and not isinstance(target_embed, PPMissingLayer))
    ):
        return
    draft_embedding = cast(Any, draft_inner).embed_tokens
    loader = DefaultModelLoader(vllm_config.load_config)
    source = DefaultModelLoader.Source(
        vllm_config.model_config.model, vllm_config.model_config.revision
    )
    embedding_names = {
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    }
    for name, weight in loader._get_weights_iterator(source):
        if name in embedding_names:
            param = draft_embedding.weight
            param.weight_loader(param, weight)
            draft.has_own_embed_tokens = True
            return
    raise ValueError("DFlash under PP requires target or draft embedding weights.")


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
        load_config=get_pp_safe_draft_load_config(get_draft_load_config(vllm_config)),
    )
    draft_vllm_config.quant_config = get_draft_quant_config(draft_vllm_config)
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    _load_pp_target_embedding(
        dflash_model, cast(nn.Module, draft_inner), target_inner, vllm_config
    )
    maybe_share_target_embed(dflash_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if (
        target_lm_head is not None
        and not getattr(dflash_model.config, "has_own_lm_head", False)
        and _should_share(
            dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
        )
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
