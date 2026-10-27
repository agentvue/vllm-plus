# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch profiles preserve hardware exclusions, precision and qualification gates."""

import hashlib
import importlib.util
import io
import json
import shlex
import sys
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
            "placement": "local",
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
    ranks = cluster.placement(candidate, "balanced-engram")
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


def test_local_tp8_pp4_keeps_each_collective_group_on_one_rig():
    ranks = cluster.placement("tp8-pp4")
    assert [set(ranks[i : i + 8]) for i in range(0, 32, 8)] == [
        {"rig9"},
        {"rig7"},
        {"rig10"},
        {"rig8"},
    ]
    tables = {1: 384006168 * 136, 14: 384016682 * 136}
    memory = cluster.engram_host_bytes("tp8-pp4", "local", tables)
    assert memory["rig9"] == cluster.math.ceil(tables[1] * 1.01)
    assert memory["rig7"] == cluster.math.ceil(tables[14] * 1.01)
    assert memory["rig10"] == memory["rig8"] == 0


@pytest.mark.parametrize("layout", [None, "balanced-engram"])
def test_launcher_cli_uses_local_by_default(layout, monkeypatch, capsys):
    args = ["deepseek_cluster.py", "serve", "--speculation", "off"]
    if layout:
        args += ["--placement", layout]
    monkeypatch.setattr(sys, "argv", args)
    cluster.main()
    command = shlex.split(capsys.readouterr().out)
    ranks = cluster.placement("tp8-pp4", layout or "local")
    ips = ",".join(cluster.NODES[rig][0] for rig in ranks)
    assert f"VLLM_RAY_ORDERED_NODE_IPS={ips}" in command
    assert "--async-scheduling" in command
    assert "--speculative-config" not in command


@pytest.mark.parametrize("reserve_gib", [15, 16])
def test_host_memory_check_rejects_shortfall_without_changing_placement(
    monkeypatch, reserve_gib
):
    available = (4 + reserve_gib) * 1024**3
    monkeypatch.setattr(
        Path, "read_text", lambda path: f"MemAvailable: {available // 1024} kB\n"
    )
    if reserve_gib < 16:
        with pytest.raises(RuntimeError, match="placement will not change"):
            cluster.check_host_memory(4 * 1024**3)
    else:
        assert cluster.check_host_memory(4 * 1024**3)["available_bytes"] == available


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


def _write_shard(path, tensors):
    header, payload = {}, bytearray()
    for name, dtype, shape, value in tensors:
        start = len(payload)
        payload.extend(value)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    return header, len(payload)


@pytest.fixture
def hybrid_source(tmp_path, monkeypatch):
    if sys.platform == "win32":
        # Windows CI lacks symlink privileges; exercise the same linked files.
        monkeypatch.setattr(
            Path, "symlink_to", lambda path, target, **kw: path.hardlink_to(target)
        )
    source, donor = tmp_path / "official", tmp_path / "download"
    source.mkdir()
    donor.mkdir()
    config = {
        "model_type": "deepseek_v41",
        "quantization_config": {
            "quant_method": "fp8",
            "expert_dtype": "fp4",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
        },
        "text_config": {
            "num_hidden_layers": 40,
            "engram_head_dim": 64,
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [3, 4],
        },
    }
    index, hashes = {"weight_map": {}, "metadata": {"total_size": 0}}, {}
    for layer, rows, name in zip((1, 14), (3, 4), cluster.ENGRAM_SHARDS):
        base = f"layers.{layer}.engram"
        for directory, packed in ((source, False), (donor, True)):
            width = 32 if packed else 64
            header, size = _write_shard(
                directory / name,
                [
                    (
                        base + ".embed.weight",
                        "U8" if packed else "F8_E4M3",
                        [rows, width],
                        bytes(rows * width),
                    ),
                    (base + ".embed.scale", "F8_E8M0", [rows, 2], b"\x7f" * rows * 2),
                    (base + ".q_weight", "BF16", [2], b"1234"),
                ],
            )
            if not packed:
                index["weight_map"].update(dict.fromkeys(header, name))
                index["metadata"]["total_size"] += size
        hashes[name] = hashlib.sha256((donor / name).read_bytes()).hexdigest()
    monkeypatch.setattr(cluster, "ENGRAM_SHARDS", hashes)
    expert_shard = "model-00003-of-00048.safetensors"
    header, size = _write_shard(
        source / expert_shard,
        [
            ("layers.0.ffn.experts.0.w1.weight", "I8", [2, 16], b"A" * 32),
            ("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", [2, 1], b"\x7f\x7f"),
        ],
    )
    index["weight_map"].update(dict.fromkeys(header, expert_shard))
    index["metadata"]["total_size"] += size
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    (source / "tokenizer.json").write_text('{"test": true}')
    return source, donor


def test_checkpoint_preflight_rejects_partial_shard(hybrid_source):
    source, _ = hybrid_source
    assert cluster.checkpoint_info(source)["engram_bytes"][14] == 4 * 66
    shard = source / next(iter(cluster.ENGRAM_SHARDS))
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        cluster.checkpoint_info(source)


def test_hybrid_assembly_preserves_experts_and_updates_table_sizes(
    hybrid_source, tmp_path
):
    source, donor = hybrid_source
    original = (source / "config.json").read_bytes()
    original_index = (source / "model.safetensors.index.json").read_bytes()
    output = tmp_path / "hybrid"
    info = cluster.prepare_engram_checkpoint(source, donor, output)
    assert info["engram_format"] == "mxfp4"
    assert info["engram_bytes"] == {1: 3 * 34, 14: 4 * 34}
    assert (source / "config.json").read_bytes() == original
    assert (source / "model.safetensors.index.json").read_bytes() == original_index
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"] == (
        json.loads(original)["quantization_config"] | cluster.ENGRAM_QUANTIZATION
    )
    assert (output / "tokenizer.json").samefile(source / "tokenizer.json")
    for name in cluster.ENGRAM_SHARDS:
        assert (output / name).samefile(donor / name)
    expert_shard = "model-00003-of-00048.safetensors"
    assert (output / expert_shard).samefile(source / expert_shard)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == info["tensor_bytes"]
    with pytest.raises(ValueError, match="destination"):
        cluster.prepare_engram_checkpoint(source, donor, output)


@pytest.mark.parametrize("failure", ["hash", "projection"])
def test_hybrid_rejects_changed_download_without_publishing(
    hybrid_source, tmp_path, monkeypatch, failure
):
    source, donor = hybrid_source
    name = next(iter(cluster.ENGRAM_SHARDS))
    path = donor / name
    path.write_bytes(path.read_bytes()[:-1] + b"X")
    if failure == "projection":
        monkeypatch.setitem(
            cluster.ENGRAM_SHARDS, name, hashlib.sha256(path.read_bytes()).hexdigest()
        )
    output = tmp_path / "hybrid"
    with pytest.raises(ValueError, match="SHA256|projection bytes"):
        cluster.prepare_engram_checkpoint(source, donor, output)
    assert not output.exists()


def test_preflight_rejects_packed_weights_with_fp8_configuration(hybrid_source):
    source, donor = hybrid_source
    for name in cluster.ENGRAM_SHARDS:
        (source / name).write_bytes((donor / name).read_bytes())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["metadata"].pop("total_size")
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Engram header disagrees"):
        cluster.checkpoint_info(source)


@pytest.mark.parametrize("failure", ["mapping", "total_size"])
def test_checkpoint_preflight_rejects_inconsistent_index(hybrid_source, failure):
    source, _ = hybrid_source
    path = source / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    if failure == "mapping":
        index["weight_map"].pop(next(iter(index["weight_map"])))
    else:
        index["metadata"]["total_size"] += 1
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="mapped tensor|total_size"):
        cluster.checkpoint_info(source)


def test_correctness_launch_keeps_precision_and_bypasses_graphs_and_prefix_cache():
    env, command = cluster.serve_command(arguments(correctness=True, speculation="off"))
    assert "--enforce-eager" in command
    assert "--no-enable-prefix-caching" in command
    assert "--enable-prefix-caching" not in command
    assert env["VLLM_DEEPSEEK_V41_DIAGNOSTICS"] == "1"
    assert command[command.index("--cpu-offload-gb") + 1] == "0"


@pytest.mark.parametrize("prefix", ["off", "on"])
@pytest.mark.parametrize("eager", [False, True])
def test_serving_separately_controls_graphs_cache_and_worker_diagnostics(prefix, eager):
    env, command = cluster.serve_command(
        arguments(speculation="off", prefix_caching=prefix, enforce_eager=eager)
    )
    assert ("--enforce-eager" in command) == eager
    assert ("--enable-prefix-caching" in command) == (prefix == "on")
    assert ("--no-enable-prefix-caching" in command) == (prefix == "off")
    for settings in [env, *json.loads(env["VLLM_RAY_NODE_ENV_VARS_JSON"]).values()]:
        assert settings["VLLM_DEEPSEEK_V41_DIAGNOSTICS"] == "0"
        assert settings["VLLM_COMPUTE_NANS_IN_LOGITS"] == "0"


def test_diagnostic_report_detects_pp_corruption_and_missing_peer(tmp_path):
    path = tmp_path / "server.log"
    packet = dict(
        event="packet",
        boundary=10,
        tp=0,
        step=1,
        tensors={"pre_mix": {"shape": [1, 4], "sha256": "one"}},
    )
    records = [
        packet | {"direction": "send"},
        packet | {"direction": "receive", "tensors": {"pre_mix": {"sha256": "two"}}},
        packet | {"direction": "send", "boundary": 30},
    ]
    path.write_text("\n".join("prefix DSV41_DIAG " + json.dumps(r) for r in records))
    report = cluster.diagnostic_report(path)
    assert len(report["packet_mismatches"]) == 1
    assert report["packet_comparisons"] == 1
    assert len(report["unpaired_packets"]) == 1
    assert not report["quality_qualified"]


def test_correctness_requests_capture_raw_tokens_and_stop_on_corruption(
    tmp_path, monkeypatch
):
    requests = []
    response = {"choices": [{"message": {"reasoning": "\ufffd"}, "token_ids": [1]}]}

    class Opener:
        def open(self, request, timeout):
            requests.append(json.loads(request.data))
            return io.StringIO(json.dumps(response))

    monkeypatch.setattr(cluster.urllib.request, "build_opener", lambda *a: Opener())
    args = arguments(output=tmp_path / "server.log")
    cluster.correctness_requests(args)
    assert len(requests) == 1
    assert requests[0]["return_token_ids"]
    assert requests[0]["logprobs"]
    assert not requests[0]["stream"]
    artifact = json.loads((tmp_path / "server.responses.json").read_text())
    assert artifact["responses"][0]["choices"][0]["token_ids"] == [1]
    assert not artifact["quality_qualified"]


def test_checkout_discovery_is_independent_of_cwd(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    (root / "vllm").mkdir(parents=True)
    (root / "tools/agentvue").mkdir(parents=True)
    (root / "tools/agentvue/check_deepseek_runtime.py").write_text("")
    monkeypatch.setattr(cluster, "__file__", str(root / "deepseek_cluster.py"))
    monkeypatch.chdir(tmp_path)
    assert cluster.checkout_root() == root
    monkeypatch.setattr(cluster, "__file__", str(tmp_path / "deepseek_cluster.py"))
    assert cluster.checkout_root(root) == root
    with pytest.raises(RuntimeError, match="--checkout"):
        cluster.checkout_root(tmp_path / "missing")


def test_empty_diagnostic_report_exposes_startup_failure(tmp_path):
    path = tmp_path / "server.log"
    path.write_text("ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'\n")
    report = cluster.diagnostic_report(path)
    assert report["status"] == "no_diagnostics_collected"
    assert report["records"] == 0
    assert report["startup_errors"]
