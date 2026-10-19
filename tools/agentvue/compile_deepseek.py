# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline SM120 code-generation check for V4.1 transport and indexer kernels.

Does not load or execute cubins and cannot qualify GPU correctness/performance.
"""

import argparse
import ast
import json
import os
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "0"

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load(path, names):
    filename = ROOT / path
    source = ast.parse(filename.read_text())
    scope = {"triton": triton, "tl": tl, "__name__": "deepseek_offline_kernels"}
    tree = ast.Module(
        body=[
            n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names
        ],
        type_ignores=[],
    )
    exec(compile(tree, str(filename), "exec"), scope)
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(args.output.resolve() / "cache")
    pp = load("vllm/models/deepseek_v41/nvidia/pp.py", {"_copy_packed_rows"})
    index = load(
        "vllm/models/deepseek_v41/nvidia/fp8_indexer.py",
        {"_score_candidates", "_sort_candidates", "_remap_candidates"},
    )
    cases = []
    for data, scales in [(576, 8), (512, 16), (128, 4)]:
        for page in (64, 128):
            for ratio in (1, 2):
                for insert in (False, True):
                    cases.append(
                        (
                            pp["_copy_packed_rows"],
                            dict(
                                cache="*u8",
                                slots="*i64",
                                positions="*i64",
                                rows="*u8",
                                num_slots="i32",
                            ),
                            dict(
                                PAGE=page,
                                STRIDE=page * (data + scales) + 576,
                                DATA=data,
                                SCALES=scales,
                                RATIO=ratio,
                                ROW_STRIDE=data + scales + 1,
                                INSERT=insert,
                                TILE=triton.next_power_of_2(data + scales + 1),
                            ),
                        )
                    )
    cases.append(
        (
            index["_sort_candidates"],
            dict(source="*i32", output="*i32"),
            dict(SOURCE_STRIDE=2048, K=2048),
        )
    )
    for dense_short in (False, True):
        width = 32768 if dense_short else 16384
        cases.append(
            (
                index["_score_candidates"],
                dict(
                    q="*fp8e4nv",
                    cache="*u8",
                    weights="*fp32",
                    block_table="*i32",
                    seq_lens="*i32",
                    candidates="*i32",
                    logits="*fp32",
                ),
                dict(
                    Q_ROW=4096,
                    Q_HEAD=128,
                    W_ROW=32,
                    TABLE_ROW=9135,
                    CACHE_PAGE=64 * 132 + 576,
                    PAGE=64,
                    K=2048,
                    BLOCK=8,
                    HEADS=32,
                    TILE=64,
                    WIDTH=width,
                    DENSE_SHORT=dense_short,
                ),
            )
        )
        cases.append(
            (
                index["_remap_candidates"],
                dict(
                    indices="*i32",
                    candidates="*i32",
                    logits="*fp32",
                    output="*i32",
                    seq_lens="*i32",
                ),
                dict(
                    OUTPUT_ROW=512,
                    K=2048,
                    BLOCK=8,
                    TOPK=512,
                    WIDTH=width,
                    DENSE_SHORT=dense_short,
                ),
            )
        )
    report = {
        "target": "sm120",
        "triton": triton.__version__,
        "gpu_execution": False,
        "cases": [],
    }
    for i, (fn, signature, constants) in enumerate(cases):
        kernel = triton.compile(
            ASTSource(fn, signature, constexprs=constants),
            target=GPUTarget("cuda", 120, 32),
            options={"num_warps": 4},
        )
        ptx = args.output / f"{i}-{fn.__name__}.ptx"
        ptx.write_text(kernel.asm["ptx"])
        report["cases"].append(
            {
                "kernel": fn.__name__,
                "constants": constants,
                "ptx": ptx.name,
                "shared_bytes": kernel.metadata.shared,
            }
        )
    report["compiled"] = len(report["cases"])
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Offline SM120 compilation passed for {len(cases)} variants; no GPU execution."
    )


if __name__ == "__main__":
    main()
