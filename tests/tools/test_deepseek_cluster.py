# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch profiles preserve hardware exclusions, precision and qualification gates."""

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/deployment/agentvue/deepseek_cluster.py"
)
if not PATH.is_file():
    pytest.skip("Local deployment launcher is not installed", allow_module_level=True)
spec = importlib.util.spec_from_file_location("deepseek_cluster", PATH)
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)


def arguments(**overrides):
    return SimpleNamespace(
        **{
            "candidate": "tp8-pp4",
            "placement": "balanced-engram",
            "profile": "latency",
            "model": cluster.MODEL,
            "port": 8000,
            "context": 584568,
            "batch_tokens": 2048,
            "candidate_scoring": False,
            "moe_backend": "auto",
            "speculation": "dspark",
            "rejection": "standard",
        }
        | overrides
    )


@pytest.mark.parametrize("candidate", cluster.CANDIDATES)
def test_placement_fits_selected_cards_and_splits_both_host_tables(candidate):
    ranks = cluster.placement(candidate)
    tp, partition = cluster.CANDIDATES[candidate]
    assert len(ranks) == tp * len(partition)
    assert all(
        count <= len(cluster.NODES[rig][2].split(","))
        for rig, count in Counter(ranks).items()
    )
    start = 0
    for stage, length in enumerate(partition):
        for layer in (1, 14):
            if start <= layer < start + length:
                counts = Counter(ranks[stage * tp : (stage + 1) * tp])
                assert len(counts) == 2
                assert set(counts.values()) == {tp // 2}
        start += length
    assert start == 40
    tables = {1: 101377628352, 14: 101380404048}
    sizes = cluster.engram_host_bytes(candidate, "balanced-engram", tables)
    assert all(size < 48 * 1024**3 for size in sizes.values())


@pytest.mark.parametrize("profile,seqs", [("latency", "1"), ("concurrency", "8")])
@pytest.mark.parametrize("speculation", ["off", "dspark"])
def test_command_preserves_precision_images_and_disables_weight_offload(
    profile, seqs, speculation
):
    env, command = cluster.serve_command(
        arguments(profile=profile, speculation=speculation)
    )

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--cpu-offload-gb") == "0"
    assert value("--kv-cache-dtype") == "fp8"
    assert value("--max-model-len") == "584568"
    assert value("--max-num-seqs") == seqs
    assert json.loads(value("--limit-mm-per-prompt")) == {"image": 2}
    attention = json.loads(value("--attention-config"))
    assert attention["indexer_kv_dtype"] == "fp8"
    assert not attention["indexer_sparse_logits"]
    assert "--skip-mm-profiling" not in command
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    if speculation == "dspark":
        cfg = json.loads(value("--speculative-config"))
        assert cfg["num_speculative_tokens"] == 5
        assert not cfg["enable_adaptive_verification"]
    else:
        assert "--speculative-config" not in command


def measured(profile, candidate, tpot, throughput):
    return {
        "profile": profile,
        "candidate": candidate,
        "checks": dict.fromkeys(
            [
                "correctness",
                "quality",
                "memory",
                "long_context",
                "images",
                "no_extra_offload",
            ],
            True,
        ),
        "runs": [
            {
                "completed": 16,
                "num_prompts": 16,
                "total_input_tokens": 16 * 8192,
                "total_output_tokens": 16 * 1024,
                "max_concurrency": 1 if profile == "latency" else 8,
                "failed": 0,
                "median_tpot_ms": tpot,
                "p95_tpot_ms": tpot * 1.1,
                "output_throughput": throughput,
            }
            for _ in range(3)
        ],
    }


def test_selection_uses_distinct_workload_objectives_and_requires_qualification():
    results = [
        measured(profile, candidate, tpot, rate)
        for profile in ["latency", "concurrency"]
        for candidate, tpot, rate in [("tp8-pp4", 10, 100), ("tp4-pp8", 15, 200)]
    ]
    selected = cluster.select_profiles(results)
    assert selected["latency"]["candidate"] == "tp8-pp4"
    assert selected["concurrency"]["candidate"] == "tp4-pp8"
    for result in results:
        result["checks"]["quality"] = False
    with pytest.raises(ValueError, match="No fully qualified"):
        cluster.select_profiles(results)


def test_selection_rejects_incomparable_workloads():
    results = [
        measured("latency", candidate, 10, 100) for candidate in ("tp8-pp4", "tp4-pp8")
    ]
    results[1]["runs"][0]["total_input_tokens"] += 1
    with pytest.raises(ValueError, match="workloads differ"):
        cluster.select_profiles(results)


@pytest.mark.parametrize("concurrency,completed", [(8, 16), (1, 0)])
def test_selection_rejects_wrong_concurrency_or_empty_measurements(
    concurrency, completed
):
    result = measured("latency", "tp8-pp4", 10, 100)
    for run in result["runs"]:
        run.update(max_concurrency=concurrency, completed=completed)
    with pytest.raises(ValueError, match="No fully qualified latency"):
        cluster.select_profiles([result])


def test_checkpoint_preflight_rejects_partial_shard(tmp_path):
    config = {
        "model_type": "deepseek_v41",
        "text_config": {
            "num_hidden_layers": 40,
            "engram_head_dim": 256,
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [100, 200],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"embed.weight": "model.safetensors"}})
    )
    header = json.dumps({"embed.weight": {"data_offsets": [0, 4]}}).encode()
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(len(header).to_bytes(8, "little") + header + b"1234")
    assert cluster.checkpoint_info(tmp_path)["engram_bytes"][14] == 200 * 264
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        cluster.checkpoint_info(tmp_path)
