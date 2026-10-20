# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check or overlay DeepSeek runtime Python files without importing CUDA."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

SHARED_FILES = (
    "config/attention.py",
    "model_executor/layers/quantization/modelopt.py",
    "model_executor/layers/quantization/mxfp4.py",
    "model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py",
    "model_executor/kernels/mhc/tilelang.py",
    "model_executor/kernels/mhc/triton.py",
    "models/deepseek_v4/nvidia/ops/o_proj.py",
    "utils/flashinfer.py",
    "utils/deep_gemm.py",
    "ray/ray_env.py",
    "v1/attention/backends/mla/indexer.py",
    "v1/attention/backends/mla/sparse_swa.py",
    "v1/core/kv_cache_utils.py",
    "v1/executor/ray_executor_v2.py",
    "v1/kv_cache_interface.py",
    "v1/worker/utils.py",
    "v1/worker/gpu/attn_utils.py",
    "v1/worker/gpu/model_runner.py",
    "v1/worker/gpu/pp_utils.py",
    "v1/worker/gpu/spec_decode/dspark/utils.py",
    "v1/worker/gpu/spec_decode/dflash/utils.py",
)
CHECKPOINT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "engram-hybrid.json",
)


def file_digest(path):
    # Treat Windows and Linux Python line endings as the same source.
    data = Path(path).read_bytes()
    if str(path).endswith(".py"):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def installed_launch_environment(environment, checkout):
    """Keep Python subprocesses on the installed package rather than the checkout."""
    environment = dict(environment)
    environment["PYTHONSAFEPATH"] = "1"
    if "PYTHONPATH" in environment:
        checkout = Path(checkout).resolve()
        entries = [
            str(Path(entry).resolve())
            for entry in environment["PYTHONPATH"].split(os.pathsep)
            if entry and Path(entry).resolve() != checkout
        ]
        if entries:
            environment["PYTHONPATH"] = os.pathsep.join(entries)
        else:
            environment.pop("PYTHONPATH")
    return environment


def verify_import_resolution(executable, cwd, environment, installed_root):
    """Check fresh-interpreter import resolution without importing vLLM or CUDA."""
    source = (
        "import importlib.machinery,json,sys; "
        "spec=importlib.machinery.PathFinder.find_spec('vllm',sys.path); "
        "print(json.dumps({'origin':spec.origin if spec else None,"
        "'safe_path':sys.flags.safe_path}))"
    )
    result = subprocess.run(
        [str(executable), "-c", source],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    report = json.loads(result.stdout)
    origin = report["origin"]
    if (
        origin is None
        or Path(origin).resolve().parent != Path(installed_root).resolve()
    ):
        raise RuntimeError(
            f"Python subprocess would import {origin}; expected {installed_root}"
        )
    return report


def source_manifest(checkout):
    root = Path(checkout).resolve() / "vllm"
    names = set(SHARED_FILES)
    names.update(
        path.relative_to(root).as_posix()
        for path in (root / "models/deepseek_v41").rglob("*.py")
    )
    return {name: file_digest(root / name) for name in sorted(names)}


def audit_installation(expected, model=None, installed_root=None):
    if installed_root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("vLLM is not installed in this interpreter")
        installed_root = Path(spec.origin).parent
    root = Path(installed_root).resolve()
    actual = {
        name: file_digest(root / name) if (root / name).is_file() else None
        for name in expected
    }
    checkpoint = {}
    if model is not None:
        checkpoint = {
            name: file_digest(Path(model) / name)
            if (Path(model) / name).is_file()
            else None
            for name in CHECKPOINT_FILES
        }
        if checkpoint["config.json"] is None or not any(
            checkpoint[name] for name in ("tokenizer.json", "tokenizer_config.json")
        ):
            raise RuntimeError("Checkpoint config/tokenizer files are missing")
    packages = {}
    for name in (
        "vllm",
        "torch",
        "flashinfer-python",
        "transformers",
        "huggingface-hub",
        "ray",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "installed_root": str(root),
        "mismatches": [name for name in expected if actual[name] != expected[name]],
        "runtime": actual,
        "checkpoint": checkpoint,
        "packages": packages,
    }


def validate_reports(reports):
    if not reports:
        raise ValueError("No runtime reports")
    failures = {
        node: report["mismatches"]
        for node, report in reports.items()
        if report["mismatches"]
    }
    if failures:
        raise ValueError(
            "Installed runtime differs from checkout: " + json.dumps(failures)
        )
    baseline = next(iter(reports.values()))
    for key in ("checkpoint", "packages"):
        if any(report[key] != baseline[key] for report in reports.values()):
            raise ValueError(f"Cluster {key} differ")


def install_runtime(checkout, installed_root=None):
    expected = source_manifest(checkout)
    report = audit_installation(expected, installed_root=installed_root)
    source = Path(checkout).resolve() / "vllm"
    destination = Path(report["installed_root"])
    for name in report["mismatches"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    return audit_installation(expected, installed_root=destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "install"))
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    expected = source_manifest(args.checkout)
    if args.action == "install":
        install_runtime(args.checkout)
    report = audit_installation(expected, args.model)
    print(json.dumps(report, indent=2))
    if report["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
