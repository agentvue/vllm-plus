# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate static E4M3 scales for a standard MLA KV cache."""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_E4M3_MAX = 448.0
SCALE_SHARD_NAME = "model-mla-fp8-kv-scales.safetensors"
REPORT_NAME = "mla-fp8-kv-calibration.json"
WORKER_EXTENSION = (
    "vllm.model_executor.layers.attention.mla_kv_calibration."
    "MLAKVCalibrationWorkerExtension"
)
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}

_BUILTIN_TEXTS = [
    (
        "A systems engineer is reviewing a distributed inference service. "
        "Explain how tensor parallel workers, pipeline stages, cache blocks, "
        "network routes, retries, and failure isolation interact. Include "
        "precise constraints, counterexamples, and a compact verification plan."
    ),
    (
        "Implement a concurrent bounded queue with cancellation, timeouts, "
        "backpressure, structured logging, and deterministic tests. Discuss "
        "memory ordering, exception safety, resource ownership, and shutdown."
    ),
    (
        "Let A be a sparse block matrix and x a partitioned vector. Derive a "
        "stable algorithm for Ax, bound its floating-point error, compare two "
        "communication schedules, and check the derivation on a small example."
    ),
    (
        "Return a JSON tool call that searches an inventory, filters by model "
        "and memory, groups devices by host, and reports missing serial numbers. "
        "The JSON must remain valid when strings contain quotes and Unicode."
    ),
    (
        "Rédige une analyse technique en français sur une panne intermittente "
        "dans un cluster de calcul. Distingue les symptômes, la cause racine, "
        "les preuves observables et les expériences qui pourraient réfuter "
        "l'hypothèse principale."
    ),
    (
        "请比较长上下文推理中的显存占用、"
        "数值精度和通信开销。给出清晰的假设、"
        "计算过程、边界情况以及可以重复执行的"
        "验证步骤。不要省略失败条件。"
    ),
    (
        "حلّل نظام استدلال موزع يحتوي على مراحل "
        "متعددة وذاكرة تخزين مؤقت مضغوطة. "
        "اشرح شروط الصحة العددية، حدود الذاكرة، "
        "وخطة اختبار قابلة للتكرار."
    ),
    (
        "Given a corrupted service log with interleaved worker messages, "
        "reconstruct the event timeline, separate the first causal exception "
        "from teardown noise, and list the minimum evidence needed for a fix."
    ),
    (
        "Design a database migration that changes a heavily used key while "
        "preserving availability. Cover dual writes, validation queries, "
        "rollback, idempotency, schema compatibility, and operational metrics."
    ),
    (
        "Write and review CUDA-style pseudocode for quantizing a tensor to an "
        "eight-bit floating-point format. Explain scale direction, saturation, "
        "rounding, vectorized loads, alignment, and reference-test tolerances."
    ),
    (
        "Analyze this mixed content exactly: NaN, infinity, -0.0, 3.1415926535, "
        "0x7f, paths/a b/c, ${VARIABLE}, <xml attr='x'>, emoji 🧪, and escaped "
        "JSON strings. Identify parsing and serialization hazards."
    ),
    (
        "Develop a deterministic needle-retrieval test for a very long document. "
        "Place distinct needles near the beginning, middle, and end; prevent "
        "position guessing; score exact answers; and diagnose false positives."
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a temporary BF16 MLA calibration pass and create a hardlinked "
            "checkpoint containing static standard-E4M3 KV-cache scales."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "Optional JSONL/plain-text corpus. JSON objects may contain prompt, "
            "text, or messages. A deterministic built-in corpus is used otherwise."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="External report path (default: <output-model>.calibration.json).",
    )
    parser.add_argument(
        "--from-report",
        type=Path,
        help="Create the sibling checkpoint from a previous report without rerunning.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--pipeline-parallel-size", type=int, default=9)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--builtin-prompt-tokens", type=int, default=8192)
    parser.add_argument("--min-calibration-tokens", type=int, default=65536)
    parser.add_argument("--generated-tokens", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.956)
    parser.add_argument("--margin", type=float, default=1.05)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_layer_count(config: dict[str, Any]) -> int:
    candidates = [config, config.get("text_config")]
    for candidate in candidates:
        if isinstance(candidate, dict):
            count = candidate.get("num_hidden_layers")
            if isinstance(count, int) and count > 0:
                return count
    raise ValueError("config.json does not contain a positive num_hidden_layers")


def _validate_source_model(model: Path, output_model: Path) -> dict[str, Any]:
    model = model.resolve()
    output_model = output_model.resolve()
    if not model.is_dir():
        raise ValueError(f"Model directory does not exist: {model}")
    if output_model.exists():
        raise ValueError(f"Output model already exists: {output_model}")
    if output_model == model or model in output_model.parents:
        raise ValueError("Output model must be a separate sibling, not inside source")
    if not output_model.parent.is_dir():
        raise ValueError(
            f"Output parent directory does not exist: {output_model.parent}"
        )

    config_path = model / "config.json"
    config = _read_json(config_path)
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ValueError("config.json has no compressed-tensors quantization_config")
    if quantization_config.get("kv_cache_scheme") is not None:
        raise ValueError("Source checkpoint already declares a kv_cache_scheme")
    if (model / SCALE_SHARD_NAME).exists():
        raise ValueError(
            f"Source contains reserved calibration shard {SCALE_SHARD_NAME}"
        )
    _model_layer_count(config)
    return config


def _load_dataset(path: Path | None) -> tuple[list[Any], str]:
    if path is None:
        return list(_BUILTIN_TEXTS), "builtin-diverse-v1"
    if not path.is_file():
        raise ValueError(f"Calibration dataset does not exist: {path}")

    records: list[Any] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = line
            if isinstance(value, str):
                records.append(value)
            elif isinstance(value, dict) and any(
                key in value for key in ("prompt", "text", "messages")
            ):
                records.append(value)
            else:
                raise ValueError(
                    f"Unsupported calibration record at {path}:{line_number}"
                )
    if not records:
        raise ValueError(f"Calibration dataset is empty: {path}")
    return records, str(path.resolve())


def _record_text(record: Any, tokenizer: Any) -> str:
    if isinstance(record, str):
        return record
    if "prompt" in record:
        return str(record["prompt"])
    if "text" in record:
        return str(record["text"])
    messages = record["messages"]
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _prepare_token_prompts(
    records: list[Any],
    tokenizer: Any,
    *,
    built_in: bool,
    max_model_len: int,
    generated_tokens: int,
    builtin_prompt_tokens: int,
    min_calibration_tokens: int,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    max_prompt_len = max_model_len - generated_tokens
    if max_prompt_len <= 0:
        raise ValueError("max_model_len must exceed generated_tokens")

    prompts: list[dict[str, list[int]]] = []
    corpus_digest = hashlib.sha256()
    token_counts: list[int] = []
    for record in records:
        text = _record_text(record, tokenizer)
        token_ids = list(tokenizer.encode(text, add_special_tokens=False))
        if not token_ids:
            continue
        if built_in:
            target = min(builtin_prompt_tokens, max_prompt_len)
            repeats = (target + len(token_ids) - 1) // len(token_ids)
            token_ids = (token_ids * repeats)[:target]
        else:
            token_ids = token_ids[:max_prompt_len]

        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is not None and len(token_ids) < max_prompt_len:
            token_ids.insert(0, int(bos_token_id))
        prompts.append({"prompt_token_ids": token_ids})
        token_counts.append(len(token_ids))
        encoded_ids = json.dumps(token_ids, separators=(",", ":")).encode()
        corpus_digest.update(len(encoded_ids).to_bytes(8, "little"))
        corpus_digest.update(encoded_ids)

    total_tokens = sum(token_counts)
    if total_tokens < min_calibration_tokens:
        raise ValueError(
            f"Calibration corpus has {total_tokens} tokens; at least "
            f"{min_calibration_tokens} are required"
        )
    return prompts, {
        "sha256_token_ids": corpus_digest.hexdigest(),
        "num_prompts": len(prompts),
        "total_prompt_tokens": total_tokens,
        "min_prompt_tokens": min(token_counts),
        "max_prompt_tokens": max(token_counts),
    }


def _round_up_float32(value: float) -> float:
    rounded = torch.tensor(value, dtype=torch.float32)
    if float(rounded.item()) < value:
        rounded = torch.nextafter(rounded, torch.tensor(math.inf))
    return float(rounded.item())


def merge_worker_reports(
    worker_reports: list[dict[str, Any]],
    *,
    num_layers: int,
    expected_replicas: int,
    margin: float,
    minimum_observed_tokens: int,
) -> list[dict[str, Any]]:
    """Merge PP layer subsets and TP replicas into one scale per layer."""
    if not math.isfinite(margin) or margin < 1.0:
        raise ValueError("Calibration margin must be finite and at least 1.0")

    replicas: dict[int, list[dict[str, Any]]] = {
        layer_index: [] for layer_index in range(num_layers)
    }
    for report in worker_reports:
        for layer_name, stats in report["layers"].items():
            match = _LAYER_INDEX_RE.search(layer_name)
            if match is None:
                raise ValueError(f"Cannot extract layer index from {layer_name}")
            layer_index = int(match.group(1))
            if layer_index >= num_layers:
                continue
            replicas[layer_index].append(
                {
                    **stats,
                    "layer_name": layer_name,
                    "hostname": report.get("hostname"),
                    "rank": report.get("rank"),
                    "tp_rank": report.get("tp_rank"),
                    "pp_rank": report.get("pp_rank"),
                }
            )

    merged: list[dict[str, Any]] = []
    for layer_index in range(num_layers):
        layer_replicas = replicas[layer_index]
        if len(layer_replicas) != expected_replicas:
            raise ValueError(
                f"Layer {layer_index} was observed on {len(layer_replicas)} "
                f"workers; expected {expected_replicas} TP replicas"
            )

        tp_ranks = {replica.get("tp_rank") for replica in layer_replicas}
        expected_tp_ranks = set(range(expected_replicas))
        if tp_ranks != expected_tp_ranks:
            raise ValueError(
                f"Layer {layer_index} was observed on TP ranks "
                f"{sorted(str(rank) for rank in tp_ranks)}; expected "
                f"{sorted(expected_tp_ranks)}"
            )
        pp_ranks = {replica.get("pp_rank") for replica in layer_replicas}
        if len(pp_ranks) != 1:
            raise ValueError(
                f"Layer {layer_index} was observed on multiple PP ranks: "
                f"{sorted(str(rank) for rank in pp_ranks)}"
            )

        bounds: list[float] = []
        token_counts: list[int] = []
        for replica in layer_replicas:
            values = (
                replica.get("latent_abs_max"),
                replica.get("rope_component_abs_max"),
                replica.get("rope_pair_radius_max"),
            )
            if any(
                value is None
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in values
            ):
                raise ValueError(
                    f"Layer {layer_index} has missing or non-finite observations"
                )
            observed_tokens = int(replica.get("observed_tokens") or 0)
            if observed_tokens < minimum_observed_tokens:
                raise ValueError(
                    f"Layer {layer_index} rank {replica.get('rank')} observed "
                    f"only {observed_tokens} cache rows; expected at least "
                    f"{minimum_observed_tokens}"
                )
            token_counts.append(observed_tokens)
            bounds.append(
                max(
                    float(replica["latent_abs_max"]),
                    float(replica["rope_pair_radius_max"]),
                )
            )

        bound = max(bounds)
        if bound <= 0.0:
            raise ValueError(f"Layer {layer_index} has a non-positive range")
        scale = _round_up_float32(max(bound * margin / FP8_E4M3_MAX, 1e-12))
        if not math.isfinite(scale):
            raise ValueError(f"Layer {layer_index} produced a non-finite scale")
        merged.append(
            {
                "layer_index": layer_index,
                "checkpoint_key": (
                    f"model.layers.{layer_index}.self_attn."
                    "mla_attn.mla_attn.k_scale"
                ),
                "latent_abs_max": max(
                    float(item["latent_abs_max"]) for item in layer_replicas
                ),
                "rope_component_abs_max": max(
                    float(item["rope_component_abs_max"])
                    for item in layer_replicas
                ),
                "rope_pair_radius_max": max(
                    float(item["rope_pair_radius_max"])
                    for item in layer_replicas
                ),
                "bound_abs_max": bound,
                "scale": scale,
                "replica_count": len(layer_replicas),
                "replica_bound_min": min(bounds),
                "replica_bound_max": max(bounds),
                "replica_relative_spread": (max(bounds) - min(bounds)) / bound,
                "observed_tokens_min": min(token_counts),
                "observed_tokens_max": max(token_counts),
                "replicas": layer_replicas,
            }
        )
    return merged


def _run_calibration(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams, __version__

    records, corpus_source = _load_dataset(args.dataset)
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        quantization="compressed-tensors",
        kv_cache_dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        distributed_executor_backend="ray",
        worker_extension_cls=WORKER_EXTENSION,
        attention_backend="TRITON_MLA_SPARSE",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enforce_eager=True,
        block_size=128,
        safetensors_load_strategy="lazy",
        enable_flashinfer_autotune=False,
        trust_remote_code=args.trust_remote_code,
        seed=0,
    )
    try:
        tokenizer = llm.get_tokenizer()
        prompts, corpus = _prepare_token_prompts(
            records,
            tokenizer,
            built_in=args.dataset is None,
            max_model_len=args.max_model_len,
            generated_tokens=args.generated_tokens,
            builtin_prompt_tokens=args.builtin_prompt_tokens,
            min_calibration_tokens=args.min_calibration_tokens,
        )
        corpus["source"] = corpus_source

        start_reports = llm.collective_rpc("start_mla_kv_calibration")
        print(
            f"Installed MLA observers on {len(start_reports)} workers; "
            f"calibrating {corpus['total_prompt_tokens']} prompt tokens.",
            flush=True,
        )
        generation_error: BaseException | None = None
        worker_reports: list[dict[str, Any]] = []
        try:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=args.generated_tokens,
                seed=0,
            )
            outputs = llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=True,
            )
            del outputs
        except BaseException as error:
            generation_error = error
        finally:
            worker_reports = llm.collective_rpc("finish_mla_kv_calibration")

        if generation_error is not None:
            raise generation_error

        layers = merge_worker_reports(
            worker_reports,
            num_layers=_model_layer_count(config),
            expected_replicas=args.tensor_parallel_size,
            margin=args.margin,
            minimum_observed_tokens=corpus["total_prompt_tokens"],
        )

        config_path = args.model / "config.json"
        index_path = args.model / "model.safetensors.index.json"
        return {
            "format_version": 1,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "vllm_version": __version__,
            "source_model": str(args.model.resolve()),
            "source_config_sha256": _sha256_file(config_path),
            "source_index_sha256": (
                _sha256_file(index_path) if index_path.is_file() else None
            ),
            "script_sha256": _sha256_file(Path(__file__)),
            "fp8_format": "e4m3fn",
            "fp8_max": FP8_E4M3_MAX,
            "scale_convention": (
                "cache_fp8 = value / scale; value = cache_fp8 * scale"
            ),
            "rope_bound": "adjacent-pair L2 radius of post-RoPE k_pe",
            "margin": args.margin,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "max_model_len": args.max_model_len,
            "generated_tokens_per_prompt": args.generated_tokens,
            "corpus": corpus,
            "worker_start_reports": start_reports,
            "workers": worker_reports,
            "layers": layers,
        }
    finally:
        engine_core = getattr(llm.llm_engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
        del llm
        gc.collect()


def _clone_file(source: Path, destination: Path, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix not in _WEIGHT_SUFFIXES:
        shutil.copy2(source, destination)
        return
    if link_mode == "hardlink":
        os.link(source, destination)
    elif link_mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def _clone_model_tree(source: Path, destination: Path, link_mode: str) -> None:
    destination.mkdir()
    for root, directories, filenames in os.walk(source):
        relative_root = Path(root).relative_to(source)
        for directory in directories:
            (destination / relative_root / directory).mkdir(exist_ok=True)
        for filename in filenames:
            _clone_file(
                Path(root) / filename,
                destination / relative_root / filename,
                link_mode,
            )


def _existing_weight_map(model: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight_map in {index_path}")
        return dict(weight_map), dict(index.get("metadata") or {})

    weight_files = sorted(model.glob("model*.safetensors"))
    if len(weight_files) != 1:
        raise ValueError(
            "Checkpoint needs model.safetensors.index.json or exactly one "
            "model*.safetensors file"
        )
    weight_file = weight_files[0]
    with safe_open(weight_file, framework="pt", device="cpu") as handle:
        weight_map = {key: weight_file.name for key in handle.keys()}
    return weight_map, {"total_size": weight_file.stat().st_size}


def create_calibrated_checkpoint(
    source_model: Path,
    output_model: Path,
    report: dict[str, Any],
    *,
    link_mode: str,
) -> None:
    """Create a sibling checkpoint with a tiny scale-only shard."""
    source_model = source_model.resolve()
    output_model = output_model.resolve()
    config = _validate_source_model(source_model, output_model)
    num_layers = _model_layer_count(config)
    report_layers = report.get("layers")
    if not isinstance(report_layers, list) or len(report_layers) != num_layers:
        raise ValueError(
            f"Report contains {len(report_layers or [])} layers; expected {num_layers}"
        )
    if report.get("source_config_sha256") != _sha256_file(
        source_model / "config.json"
    ):
        raise ValueError("Report was produced from a different config.json")
    source_index = source_model / "model.safetensors.index.json"
    report_index_hash = report.get("source_index_sha256")
    if report_index_hash is not None and (
        not source_index.is_file()
        or report_index_hash != _sha256_file(source_index)
    ):
        raise ValueError("Report was produced from a different safetensors index")

    scale_tensors: dict[str, torch.Tensor] = {}
    for expected_index, layer in enumerate(report_layers):
        if int(layer.get("layer_index", -1)) != expected_index:
            raise ValueError("Calibration report layer order is incomplete")
        key = layer.get("checkpoint_key")
        expected_key = (
            f"model.layers.{expected_index}.self_attn."
            "mla_attn.mla_attn.k_scale"
        )
        if key != expected_key:
            raise ValueError(f"Unexpected scale tensor key: {key}")
        scale = float(layer["scale"])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Layer {expected_index} has invalid scale {scale}")
        scale_tensors[key] = torch.tensor([scale], dtype=torch.float32)

    weight_map, metadata = _existing_weight_map(source_model)
    duplicate_keys = sorted(set(scale_tensors).intersection(weight_map))
    if duplicate_keys:
        raise ValueError(
            f"Checkpoint already contains FP8 MLA scales: {duplicate_keys}"
        )

    temporary = output_model.with_name(f".{output_model.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"Temporary output already exists: {temporary}")
    try:
        _clone_model_tree(source_model, temporary, link_mode)
        save_file(
            scale_tensors,
            temporary / SCALE_SHARD_NAME,
            metadata={"format": "pt"},
        )
        with safe_open(
            temporary / SCALE_SHARD_NAME,
            framework="pt",
            device="cpu",
        ) as handle:
            if set(handle.keys()) != set(scale_tensors):
                raise RuntimeError("Scale shard verification failed")

        for key in scale_tensors:
            weight_map[key] = SCALE_SHARD_NAME
        total_size = int(metadata.get("total_size", 0))
        metadata["total_size"] = total_size + 4 * len(scale_tensors)
        _write_json_atomic(
            temporary / "model.safetensors.index.json",
            {"metadata": metadata, "weight_map": weight_map},
        )

        quantization_config = config["quantization_config"]
        quantization_config["kv_cache_scheme"] = {
            "dynamic": False,
            "num_bits": 8,
            "strategy": "tensor",
            "symmetric": True,
            "type": "float",
        }
        _write_json_atomic(temporary / "config.json", config)

        report["output_model"] = str(output_model)
        report["output_config_sha256"] = _sha256_file(temporary / "config.json")
        report["output_index_sha256"] = _sha256_file(
            temporary / "model.safetensors.index.json"
        )
        report["scale_shard_sha256"] = _sha256_file(
            temporary / SCALE_SHARD_NAME
        )
        _write_json_atomic(temporary / REPORT_NAME, report)
        os.replace(temporary, output_model)
    except BaseException:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    args = parse_args()
    args.model = args.model.resolve()
    args.output_model = args.output_model.resolve()
    if args.tensor_parallel_size <= 0 or args.pipeline_parallel_size <= 0:
        raise ValueError("Parallel sizes must be positive")
    if args.max_model_len <= 0 or args.generated_tokens <= 0:
        raise ValueError("Model length and generated token count must be positive")
    if args.min_calibration_tokens <= 0 or args.builtin_prompt_tokens <= 0:
        raise ValueError("Calibration token counts must be positive")
    config = _validate_source_model(args.model, args.output_model)
    if (
        args.link_mode == "hardlink"
        and args.model.stat().st_dev != args.output_model.parent.stat().st_dev
    ):
        raise ValueError("Hardlink output must be on the source model filesystem")
    report_path = (
        args.report.resolve()
        if args.report is not None
        else args.output_model.with_name(f"{args.output_model.name}.calibration.json")
    )
    if report_path == args.output_model or args.output_model in report_path.parents:
        raise ValueError("External report path must be outside the output model")

    if args.from_report is not None:
        report = _read_json(args.from_report.resolve())
    else:
        report = _run_calibration(args, config)
        _write_json_atomic(report_path, report)
        print(f"Calibration report written to {report_path}", flush=True)

    create_calibrated_checkpoint(
        args.model,
        args.output_model,
        report,
        link_mode=args.link_mode,
    )
    _write_json_atomic(report_path, report)
    print(f"Calibrated checkpoint created at {args.output_model}", flush=True)
    print(
        "Run it with --kv-cache-dtype fp8, then complete retrieval and "
        "throughput validation before production use.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
