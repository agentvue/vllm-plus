# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict repeated-run performance gate for vllm bench serve JSON results."""

import argparse
import json
import statistics
from pathlib import Path


def read_runs(directory):
    runs = [
        json.loads(path.read_text())
        for path in sorted(directory.glob("*.json"))
        if path.stem != "0"
    ]
    if len(runs) < 5:
        raise ValueError(
            "At least five measured runs are required, excluding warmup 0.json"
        )
    for run in runs:
        if (
            run["completed"] <= 0
            or run["completed"] != run["num_prompts"]
            or run["failed"]
            or run.get("errors")
        ):
            raise ValueError("A benchmark run has failed requests or no completions")
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline, candidate = read_runs(args.baseline), read_runs(args.candidate)
    signature = (
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "request_rate",
        "max_concurrency",
    )
    expected = tuple(baseline[0][key] for key in signature)
    if any(
        tuple(run[key] for key in signature) != expected for run in baseline + candidate
    ):
        raise ValueError(
            "Workloads differ: freeze prompts, outputs, traffic, and concurrency"
        )
    metrics = {
        "output_throughput": True,
        "request_throughput": True,
        "median_ttft_ms": False,
        "p99_ttft_ms": False,
        "median_tpot_ms": False,
        "p99_tpot_ms": False,
        "median_e2el_ms": False,
        "p99_e2el_ms": False,
    }
    failures = []
    for metric, higher_is_better in metrics.items():
        before = statistics.median(run[metric] for run in baseline)
        after = statistics.median(run[metric] for run in candidate)
        passed = after >= before if higher_is_better else after <= before
        print(f"{metric}: baseline={before:.6g}, candidate={after:.6g}, pass={passed}")
        if not passed:
            failures.append(metric)
    if failures:
        raise SystemExit("Strict performance gate failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
