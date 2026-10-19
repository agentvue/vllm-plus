# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute unchanged V4.1 Triton kernels in the CPU interpreter.

Checks byte addressing and FP8 arithmetic, not GPU compilation, CUDA graphs,
distributed execution, timing or native extension ABI. Requires Triton and torch.
"""

import ast
import os
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def kernels(path, names):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    selected = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    scope = {"torch": torch, "triton": triton, "tl": tl}
    exec(compile(selected, str(source), "exec"), scope)
    return scope


def check_transport(copy_rows, data, scales, ratio):
    width = data + scales
    page, target_page = 128, 64
    stride, target_stride = page * width + 256, target_page * width + 512
    raw = (torch.arange(stride * 3) % 251).to(torch.uint8)
    target_raw = torch.full((target_stride * 5,), 253, dtype=torch.uint8)
    cache = raw.as_strided((3, page, width), (stride, width, 1))
    target = target_raw.as_strided((5, target_page, width), (target_stride, width, 1))
    slots = torch.tensor([0, 127, -1, 128, 129, 256, 257, 258])
    destinations = torch.tensor([63, 64, 65, 128, -1, 191, 192, 193])
    positions = torch.tensor([1, 2, 3, 4, 5, 127, 128, 129])
    packed = torch.empty((8, width + 1), dtype=torch.uint8)
    for _ in range(2):
        target_raw.fill_(253)
        copy_rows(cache, slots, positions, packed, data, ratio, insert=False)
        copy_rows(target, destinations, positions, packed, data, ratio, insert=True)
        expected = torch.full_like(target_raw, 253)
        for source, destination, position in zip(slots, destinations, positions):
            source, destination, position = int(source), int(destination), int(position)
            if min(source, destination) < 0 or (position + 1) % ratio:
                continue
            for count, src, dst in (
                (data, source % page * data, destination % target_page * data),
                (
                    scales,
                    page * data + source % page * scales,
                    target_page * data + destination % target_page * scales,
                ),
            ):
                src += source // page * stride
                dst += destination // target_page * target_stride
                expected[dst : dst + count] = raw[src : src + count]
        assert torch.equal(target_raw, expected)
        positions.add_(1)
        slots = slots.roll(1)
        destinations = destinations.roll(1)


def check_scores(score, remap, dense_short):
    torch.manual_seed(42)
    rows, page, pages, context = 2, 64, 17, 1025
    q = torch.randn((rows + 1, 32, 128)).to(torch.float8_e4m3fn)[1:]
    keys = torch.randn((pages, page, 128)).to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randint(-3, 4, (pages, page)).float())
    raw = torch.empty((pages, page * 132 + 256), dtype=torch.uint8)
    raw[:, : page * 128].copy_(keys.view(torch.uint8).reshape(pages, -1))
    raw[:, page * 128 : page * 132].copy_(scales.view(torch.uint8).reshape(pages, -1))
    cache = raw.as_strided((pages, page, 132), (raw.stride(0), 132, 1))
    table = torch.arange(pages, dtype=torch.int32).repeat(rows, 1)
    ends = torch.tensor([context, context - 5], dtype=torch.int32)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32)
    candidates[:, :129] = torch.randperm(129).to(torch.int32)
    weights = torch.randn((rows, 32))
    ordered = torch.empty_like(candidates)
    width = 32768 if dense_short else 16384
    scores = torch.empty((rows, width), dtype=torch.float32)
    score(
        q,
        cache,
        weights,
        table,
        ends,
        candidates,
        ordered,
        scores,
        8,
        dense_short_context=dense_short,
    )
    selected = scores.topk(512, dim=1).indices.to(torch.int32)
    output = torch.empty_like(selected)
    remap[(rows,)](
        selected, ordered, scores, output, ends, 512, 2048, 8, 512, width, dense_short
    )
    for row in range(rows):
        if dense_short:
            tokens = torch.arange(width)
            valid = (tokens < ends[row]) & torch.isin(tokens // 8, candidates[row])
        else:
            tokens = ordered[row].long().repeat_interleave(8) * 8 + torch.arange(
                8
            ).repeat(2048)
            valid = tokens < ends[row]
        safe = tokens.clamp(0, context - 1)
        dot = q[row].float() @ keys.reshape(-1, 128).float()[safe].T
        expected = (dot.relu() * weights[row, :, None]).sum(0) * scales.flatten()[safe]
        expected.masked_fill_(~valid, -float("inf"))
        torch.testing.assert_close(scores[row], expected, rtol=1e-4, atol=2e-3)
        assert torch.equal(output[row].long(), tokens[selected[row].long()])


def main():
    pp = kernels(
        "vllm/models/deepseek_v41/nvidia/pp.py",
        {"_copy_packed_rows", "copy_packed_rows"},
    )
    for data, scales in [(576, 8), (512, 16), (128, 4)]:
        for ratio in (1, 2):
            check_transport(pp["copy_packed_rows"], data, scales, ratio)
    indexer = kernels(
        "vllm/models/deepseek_v41/nvidia/fp8_indexer.py",
        {
            "_sort_candidates",
            "_score_candidates",
            "_remap_candidates",
            "candidate_scores",
        },
    )
    for dense_short in (False, True):
        check_scores(
            indexer["candidate_scores"], indexer["_remap_candidates"], dense_short
        )
    print(
        "Passed: 12 packed-cache updates and 4 FP8 score/remap rows "
        "in the CPU interpreter."
    )
    print("GPU compilation, CUDA graphs, accuracy and performance remain unqualified.")


if __name__ == "__main__":
    main()
