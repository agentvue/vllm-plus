# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.calibrate_mla_fp8_kv import (
    SCALE_SHARD_NAME,
    _sha256_file,
    create_calibrated_checkpoint,
    merge_worker_reports,
)
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.attention.mla_kv_calibration import (
    MLAKVCalibrationWorkerExtension,
)


class _DummyImpl:
    def __init__(self) -> None:
        self.calls = 0

    def do_kv_cache_update(self, *args, **kwargs):
        self.calls += 1
        return "updated"


class _DummyWorker(MLAKVCalibrationWorkerExtension):
    rank = 3

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def get_model(self) -> torch.nn.Module:
        return self.model


def _dummy_mla_layer() -> MLAAttention:
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.2.self_attn.attn"
    layer.impl = _DummyImpl()
    return layer


def test_worker_extension_observes_only_cache_writes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.mla_kv_calibration."
        "get_tensor_model_parallel_rank",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.mla_kv_calibration."
        "get_pp_group",
        lambda: type("PPGroup", (), {"rank_in_group": 4})(),
    )
    model = torch.nn.Module()
    model.mla = _dummy_mla_layer()
    worker = _DummyWorker(model)

    start = worker.start_mla_kv_calibration()
    assert start["num_layers"] == 1

    kv_c = torch.tensor([[3.0, 4.0], [1000.0, 1000.0]])
    k_pe = torch.tensor([[[3.0, 4.0, 0.0, 0.0]], [[1000.0] * 4]])
    slots = torch.tensor([7, -1])
    assert (
        model.mla.impl.do_kv_cache_update(
            kv_c,
            k_pe,
            torch.empty(0),
            slots,
            "bfloat16",
            torch.tensor(1.0),
        )
        == "updated"
    )
    assert (
        model.mla.impl.do_kv_cache_update(
            torch.tensor([[6.0, 2.0]]),
            torch.tensor([[[5.0, 12.0, 0.0, 0.0]]]),
            torch.empty(0),
            torch.tensor([8]),
            "bfloat16",
            torch.tensor(1.0),
        )
        == "updated"
    )

    result = worker.finish_mla_kv_calibration()
    stats = result["layers"][model.mla.layer_name]
    assert stats["observed_tokens"] == 2
    assert stats["latent_abs_max"] == 6.0
    assert stats["rope_component_abs_max"] == 12.0
    assert stats["rope_pair_radius_max"] == 13.0
    assert result["tp_rank"] == 1
    assert result["pp_rank"] == 4
    assert model.mla.impl.calls == 2
    assert not hasattr(worker, "_mla_kv_calibration_state")


def _worker_report(layer_index: int, rank: int, bound: float) -> dict:
    return {
        "hostname": f"host-{rank // 4}",
        "rank": rank,
        "tp_rank": rank % 4,
        "pp_rank": rank // 4,
        "layers": {
            f"model.layers.{layer_index}.self_attn.attn": {
                "calls": 2,
                "observed_tokens": 100,
                "latent_abs_max": bound - 1.0,
                "rope_component_abs_max": bound - 0.5,
                "rope_pair_radius_max": bound,
            }
        },
    }


def test_merge_worker_reports_uses_tp_max_and_rope_bound():
    reports = []
    for layer_index in range(2):
        reports.extend(
            _worker_report(layer_index, layer_index * 4 + rank, 10.0 + rank)
            for rank in range(4)
        )

    layers = merge_worker_reports(
        reports,
        num_layers=2,
        expected_replicas=4,
        margin=1.05,
        minimum_observed_tokens=100,
    )

    assert len(layers) == 2
    assert layers[0]["bound_abs_max"] == 13.0
    assert layers[0]["replica_count"] == 4
    assert layers[0]["checkpoint_key"].endswith(
        ".self_attn.mla_attn.mla_attn.k_scale"
    )
    required_scale = 13.0 * 1.05 / 448.0
    assert layers[0]["scale"] >= required_scale
    assert math.isfinite(layers[0]["scale"])


def test_merge_worker_reports_rejects_missing_tp_replica():
    reports = [_worker_report(0, rank, 10.0) for rank in range(3)]
    with pytest.raises(ValueError, match="expected 4 TP replicas"):
        merge_worker_reports(
            reports,
            num_layers=1,
            expected_replicas=4,
            margin=1.05,
            minimum_observed_tokens=100,
        )


def test_merge_worker_reports_rejects_duplicate_tp_rank():
    reports = [_worker_report(0, rank, 10.0) for rank in range(4)]
    reports[-1]["tp_rank"] = 2
    with pytest.raises(ValueError, match="expected.*0, 1, 2, 3"):
        merge_worker_reports(
            reports,
            num_layers=1,
            expected_replicas=4,
            margin=1.05,
            minimum_observed_tokens=100,
        )


def _write_source_checkpoint(path: Path) -> dict:
    path.mkdir()
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 2,
        "quantization_config": {
            "format": "pack-quantized",
            "config_groups": {},
            "quant_method": "compressed-tensors",
        },
    }
    (path / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    weight_file = path / "model-00001-of-00001.safetensors"
    save_file(
        {"model.embed_tokens.weight": torch.ones(1)},
        weight_file,
        metadata={"format": "pt"},
    )
    index = {
        "metadata": {"total_size": 4},
        "weight_map": {
            "model.embed_tokens.weight": weight_file.name,
        },
    }
    (path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    return config


def test_create_calibrated_checkpoint_adds_internal_scale_keys(tmp_path: Path):
    source = tmp_path / "source"
    config = _write_source_checkpoint(source)
    output = tmp_path / "output"
    layers = [
        {
            "layer_index": layer_index,
            "checkpoint_key": (
                f"model.layers.{layer_index}.self_attn."
                "mla_attn.mla_attn.k_scale"
            ),
            "scale": 0.125 + layer_index,
        }
        for layer_index in range(2)
    ]
    report = {
        "source_config_sha256": _sha256_file(source / "config.json"),
        "layers": layers,
    }

    create_calibrated_checkpoint(
        source,
        output,
        report,
        link_mode="hardlink",
    )

    output_config = json.loads((output / "config.json").read_text())
    scheme = output_config["quantization_config"]["kv_cache_scheme"]
    assert scheme == {
        "dynamic": False,
        "num_bits": 8,
        "strategy": "tensor",
        "symmetric": True,
        "type": "float",
    }
    assert "kv_cache_scheme" not in config["quantization_config"]
    assert os.path.samefile(
        source / "model-00001-of-00001.safetensors",
        output / "model-00001-of-00001.safetensors",
    )

    output_index = json.loads(
        (output / "model.safetensors.index.json").read_text()
    )
    for layer in layers:
        assert output_index["weight_map"][layer["checkpoint_key"]] == (
            SCALE_SHARD_NAME
        )
    with safe_open(
        output / SCALE_SHARD_NAME,
        framework="pt",
        device="cpu",
    ) as handle:
        assert set(handle.keys()) == {
            layer["checkpoint_key"] for layer in layers
        }
        assert handle.get_tensor(layers[0]["checkpoint_key"]).item() == 0.125
