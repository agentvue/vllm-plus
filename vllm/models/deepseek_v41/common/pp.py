# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependencies crossing a DeepSeek V4.1 pipeline boundary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SharedCacheDependency:
    source: int
    index_source: int
    compress_ratio: int
    candidates: bool


def shared_cache_dependency(config, boundary: int) -> SharedCacheDependency | None:
    """Return state the layer at ``boundary`` needs from an earlier stage."""
    if not 0 < boundary < config.num_hidden_layers:
        return None
    ratio = config.compress_ratios[boundary]
    if ratio == 0:
        return None
    sources = [s for s in config.kv_source_layer_ids if s <= boundary]
    indices = [s for s in config.index_source_layer_ids if s <= boundary]
    if not sources or not indices:
        raise ValueError("Compressed layers require preceding KV and index sources")
    source, index_source = max(sources), max(indices)
    if source == boundary:
        return None
    if config.compress_ratios[source] != ratio:
        raise ValueError("KV consumers must use their source's compression ratio")
    return SharedCacheDependency(
        source,
        index_source,
        ratio,
        0 <= getattr(config, "candidate_source_layer_id", -1) < boundary,
    )
