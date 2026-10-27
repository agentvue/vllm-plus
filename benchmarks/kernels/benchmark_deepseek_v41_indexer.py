# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare candidate-score construction with dense FP8 scoring plus masking.

This measures scoring only, including candidate sorting/masking, but not top-k,
PP communication, or model throughput. Use the serving sweep to select profiles.
"""

import argparse
import importlib.metadata
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
)
from vllm.models.deepseek_v41.nvidia.fp8_indexer import candidate_scores
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)


def measure(rows, context, repeat):
    torch.manual_seed(42)
    page = 64
    pages = (context + page - 1) // page
    q = torch.randn((rows, 32, 128), device="cuda").to(torch.float8_e4m3fn)
    keys = torch.randn((pages, page, 128), device="cuda").to(torch.float8_e4m3fn)
    cache = torch.empty((pages, page, 132), dtype=torch.uint8, device="cuda")
    raw = cache.view(pages, -1)
    raw[:, : page * 128].copy_(keys.view(torch.uint8).reshape(pages, -1))
    scales = torch.ones((pages, page), dtype=torch.float32, device="cuda")
    raw[:, page * 128 :].copy_(scales.view(torch.uint8).reshape(pages, -1))
    weights = torch.randn((rows, 32), device="cuda")
    table = torch.arange(pages, dtype=torch.int32, device="cuda").repeat(rows, 1)
    ends = torch.full((rows, 1), context, dtype=torch.int32, device="cuda")
    candidates = torch.stack(
        [torch.randperm((context + 7) // 8, device="cuda")[:2048] for _ in range(rows)]
    ).to(torch.int32)
    ordered = torch.empty_like(candidates)
    compact = torch.empty((rows, 16384), dtype=torch.float32, device="cuda")
    metadata = get_paged_mqa_logits_metadata(
        ends, page, torch.cuda.get_device_properties(0).multi_processor_count
    )

    def dense():
        logits = fp8_fp4_paged_mqa_logits(
            (q[:, None], None),
            cache.unsqueeze(-2),
            weights,
            ends,
            table,
            metadata,
            max_model_len=context,
            clean_logits=False,
        )
        apply_candidate_mask(logits, None, ends.flatten(), candidates, 8)
        return logits

    def sparse():
        candidate_scores(
            q, cache, weights, table, ends, candidates, ordered, compact, 8
        )

    started = time.perf_counter()
    baseline = dense()
    sparse()
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - started
    indices = ordered.long().repeat_interleave(8, dim=1) * 8
    indices += torch.arange(8, device="cuda").repeat(2048)
    valid = indices < context
    expected = baseline.gather(1, indices.clamp_max(context - 1))
    expected.masked_fill_(~valid, -float("inf"))
    torch.testing.assert_close(compact, expected, rtol=1e-4, atol=2e-3)
    results = []
    for label, fn, tokens in [
        ("dense_masked", dense, context),
        ("candidate", sparse, 16384),
    ]:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        us = (
            statistics.median(
                bench_gpu_time_with_cupti(
                    fn,
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                    dry_run_iters=5,
                    repeat_iters=repeat,
                )
            )
            * 1000
        )
        results.append(
            {
                "backend": label,
                "rows": rows,
                "context": context,
                "us": us,
                "scoring_tflops": 2 * rows * 32 * 128 * tokens / (us * 1e6),
            }
        )
    return {"correctness": "passed", "first_calls_s": first_call_s, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 48])
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[32768, 131072, 584568]
    )
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("This comparison requires an SM120 GPU")
    if min(args.contexts) < 16384 or min(args.rows) < 1:
        parser.error("Use positive row counts and contexts of at least 16384")
    report = {
        "gpu": torch.cuda.get_device_name(),
        "dtype": "FP8 Q/K, FP32 scores",
        "scope": "scoring only; no top-k or PP",
        "torch": torch.__version__,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "cases": [
            measure(rows, context, args.repeat)
            for context in args.contexts
            for rows in args.rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
