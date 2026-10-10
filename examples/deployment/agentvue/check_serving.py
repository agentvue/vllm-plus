# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concurrent PP correctness, cancellation, slot reuse, and repeated-prefix smoke.

Save baseline results first; use --reference-json on identical candidate settings.
Exact greedy token agreement is a smoke gate; corpus accuracy and performance
are separate gates. Run through the cluster's uv-managed virtual environment.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def request(url, payload=None):
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    return urlopen(
        Request(
            url, data=json.dumps(payload).encode() if payload else None, headers=headers
        ),
        timeout=600,
    )


def complete(base_url, model, item):
    payload = {
        "model": model,
        "prompt": item["prompt"],
        "max_tokens": item["max_tokens"],
        "temperature": 0,
        "seed": 42,
        "return_token_ids": True,
    }
    started = time.perf_counter()
    with request(base_url + "/v1/completions", payload) as response:
        result = json.load(response)
    choice = result["choices"][0]
    return {
        **item,
        "text": choice["text"],
        "token_ids": choice.get("token_ids"),
        "finish_reason": choice["finish_reason"],
        "elapsed_s": time.perf_counter() - started,
    }


def cancel(base_url, model, item):
    payload = {
        "model": model,
        "prompt": item["prompt"],
        "max_tokens": 512,
        "temperature": 0,
        "stream": True,
        "seed": 42,
    }
    with request(base_url + "/v1/completions", payload) as response:
        for line in response:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk["choices"][0].get("text"):
                    # Closing the streaming response triggers the server's abort path.
                    return True
    raise RuntimeError("Cancellation request finished before producing a stream chunk")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="agentvue")
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument(
        "--prompts-json",
        type=Path,
        help="JSON list of {prompt,max_tokens} objects; overrides defaults",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.concurrency < 1 or args.rounds < 1:
        parser.error("concurrency and rounds must be positive")
    reference = (
        json.loads(args.reference_json.read_text()) if args.reference_json else None
    )
    prompts = (
        json.loads(args.prompts_json.read_text())
        if args.prompts_json
        else reference["prompts"]
        if reference
        else [
            {
                "prompt": "Summarize this passage: "
                + ("The river crosses the city. " * n),
                "max_tokens": tokens,
            }
            for n, tokens in [(1, 8), (8, 16), (64, 32), (256, 64)]
        ]
    )
    base_url = args.base_url.rstrip("/")
    with request(base_url + "/health") as response:
        assert response.status == 200
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for _ in range(args.rounds):
            # Interleave aborts with complete requests to cause slot release and reuse.
            pending = []
            for item in prompts * 2:
                pending.append(pool.submit(complete, base_url, args.model, item))
                pending.append(pool.submit(cancel, base_url, args.model, item))
            for index, future in enumerate(pending):
                value = future.result()
                if index % 2 == 0:
                    results.append(value)
        # Repeat the exact prefixes after cancellations and batch reordering.
        reused = list(
            pool.map(
                lambda item: complete(base_url, args.model, item), reversed(prompts)
            )
        )
    results.extend(reused)
    with request(base_url + "/metrics") as response:
        metrics = response.read().decode()
    mismatches = []
    expected = reference["results"] if reference else results[: len(prompts)]
    by_prompt = {(item["prompt"], item["max_tokens"]): item for item in expected}
    for index, actual in enumerate(results):
        original = by_prompt[(actual["prompt"], actual["max_tokens"])]
        if (
            actual["text"] != original["text"]
            or actual["token_ids"] != original["token_ids"]
        ):
            mismatches.append(index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "prompts": prompts,
                "results": results,
                "mismatches": mismatches,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "completed": len(results),
                "cancelled": len(prompts) * 2 * args.rounds,
                "mismatches": mismatches,
            }
        )
    )
    if mismatches:
        raise SystemExit(
            "Greedy results changed; inspect saved outputs and run accuracy evaluation"
        )


if __name__ == "__main__":
    main()
