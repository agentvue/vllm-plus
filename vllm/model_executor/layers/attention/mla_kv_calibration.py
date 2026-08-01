# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed observer used by the offline MLA FP8 KV calibrator."""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any

import torch

from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
)
from vllm.model_executor.layers.attention.mla_attention import MLAAttention


class MLAKVCalibrationWorkerExtension:
    """Observe the exact BF16 rows written to MLA KV caches."""

    def start_mla_kv_calibration(self) -> dict[str, Any]:
        """Install cache-write observers on every local MLA layer."""
        if hasattr(self, "_mla_kv_calibration_state"):
            raise RuntimeError("MLA KV calibration is already active")

        model = self.get_model()  # type: ignore[attr-defined]
        layers: dict[str, dict[str, Any]] = {}
        restore: list[tuple[Any, Callable[..., Any]]] = []

        for module in model.modules():
            if not isinstance(module, MLAAttention):
                continue

            layer_name = module.layer_name
            state: dict[str, Any] = {
                "calls": 0,
                "observed_tokens": None,
                "latent_abs_max": None,
                "rope_component_abs_max": None,
                "rope_pair_radius_max": None,
            }
            layers[layer_name] = state
            impl = module.impl
            original = impl.do_kv_cache_update

            def wrapped_cache_update(
                *args: Any,
                _original: Callable[..., Any] = original,
                _state: dict[str, Any] = state,
                _layer_name: str = layer_name,
                **kwargs: Any,
            ) -> Any:
                kv_c_normed = kwargs.get("kv_c_normed")
                k_pe = kwargs.get("k_pe")
                slot_mapping = kwargs.get("slot_mapping")
                if kv_c_normed is None and len(args) > 0:
                    kv_c_normed = args[0]
                if k_pe is None and len(args) > 1:
                    k_pe = args[1]
                if slot_mapping is None and len(args) > 3:
                    slot_mapping = args[3]

                self._observe_mla_cache_rows(
                    _layer_name,
                    _state,
                    kv_c_normed,
                    k_pe,
                    slot_mapping,
                )
                return _original(*args, **kwargs)

            setattr(impl, "do_kv_cache_update", wrapped_cache_update)
            restore.append((impl, original))

        if not layers:
            raise RuntimeError("No MLAAttention layers were found on this worker")

        self._mla_kv_calibration_state = {
            "layers": layers,
            "restore": restore,
        }
        return {
            "hostname": socket.gethostname(),
            "rank": getattr(self, "rank", getattr(self, "rpc_rank", None)),
            "num_layers": len(layers),
            "device": self._device_info(),
        }

    @staticmethod
    def _observe_mla_cache_rows(
        layer_name: str,
        state: dict[str, Any],
        kv_c_normed: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        if kv_c_normed is None or k_pe is None or slot_mapping is None:
            return
        if not isinstance(slot_mapping, torch.Tensor):
            raise TypeError(
                f"{layer_name}: expected tensor slot_mapping, "
                f"got {type(slot_mapping)}"
            )

        num_rows = min(
            kv_c_normed.shape[0],
            k_pe.shape[0],
            slot_mapping.numel(),
        )
        if num_rows == 0:
            return

        valid = slot_mapping.reshape(-1)[:num_rows] >= 0
        row_mask = valid.reshape(num_rows, *([1] * (kv_c_normed.ndim - 1)))
        latent = kv_c_normed[:num_rows].detach().float().abs()
        latent_abs_max = latent.masked_fill(~row_mask, 0.0).amax()

        rope = k_pe[:num_rows].detach().float()
        if rope.ndim == 3 and rope.shape[-2] == 1:
            rope = rope.squeeze(-2)
        if rope.ndim != 2 or rope.shape[-1] % 2:
            raise ValueError(
                f"{layer_name}: expected post-RoPE k_pe shape [T, 1, 2N], "
                f"got {tuple(k_pe.shape)}"
            )
        rope_mask = valid.reshape(num_rows, 1)
        rope_component_abs_max = rope.abs().masked_fill(~rope_mask, 0.0).amax()
        rope_pair_radius = torch.hypot(rope[..., 0::2], rope[..., 1::2])
        rope_pair_radius_max = rope_pair_radius.masked_fill(
            ~rope_mask, 0.0
        ).amax()

        MLAKVCalibrationWorkerExtension._update_max(
            state, "latent_abs_max", latent_abs_max
        )
        MLAKVCalibrationWorkerExtension._update_max(
            state, "rope_component_abs_max", rope_component_abs_max
        )
        MLAKVCalibrationWorkerExtension._update_max(
            state, "rope_pair_radius_max", rope_pair_radius_max
        )
        valid_count = valid.sum(dtype=torch.int64)
        if state["observed_tokens"] is None:
            state["observed_tokens"] = valid_count
        else:
            state["observed_tokens"].add_(valid_count)
        state["calls"] += 1

    @staticmethod
    def _update_max(
        state: dict[str, Any], name: str, value: torch.Tensor
    ) -> None:
        value = value.detach()
        if state[name] is None:
            state[name] = value
        else:
            state[name].copy_(torch.maximum(state[name], value))

    def finish_mla_kv_calibration(self) -> dict[str, Any]:
        """Remove observers and return plain per-layer statistics."""
        calibration = getattr(self, "_mla_kv_calibration_state", None)
        if calibration is None:
            raise RuntimeError("MLA KV calibration is not active")

        for impl, original in calibration["restore"]:
            setattr(impl, "do_kv_cache_update", original)

        result_layers: dict[str, dict[str, int | float | None]] = {}
        for layer_name, state in calibration["layers"].items():
            result_layers[layer_name] = {
                "calls": state["calls"],
                "observed_tokens": self._scalar_item(state["observed_tokens"], int),
                "latent_abs_max": self._scalar_item(
                    state["latent_abs_max"], float
                ),
                "rope_component_abs_max": self._scalar_item(
                    state["rope_component_abs_max"], float
                ),
                "rope_pair_radius_max": self._scalar_item(
                    state["rope_pair_radius_max"], float
                ),
            }

        del self._mla_kv_calibration_state
        return {
            "hostname": socket.gethostname(),
            "rank": getattr(self, "rank", getattr(self, "rpc_rank", None)),
            "tp_rank": get_tensor_model_parallel_rank(),
            "pp_rank": get_pp_group().rank_in_group,
            "device": self._device_info(),
            "layers": result_layers,
        }

    @staticmethod
    def _scalar_item(value: torch.Tensor | None, cast: type) -> Any:
        if value is None:
            return None
        return cast(value.item())

    @staticmethod
    def _device_info() -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {"type": "cpu"}
        index = torch.cuda.current_device()
        return {
            "type": "cuda",
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        }
