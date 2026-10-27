# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate scoring matches the original FP8 indexer's masked FP32 scores."""

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.fp8_indexer import candidate_scores

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="requires consumer Blackwell",
)


@pytest.mark.parametrize("context", [32769, 131073])
@pytest.mark.parametrize("rows", [1, 6])
@pytest.mark.parametrize("dense_short", [False, True])
def test_fp8_candidate_scores_preserve_causality_scaling_and_negative_weights(
    context, rows, dense_short, monkeypatch
):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(42)
    page = 64
    pages = (context + page - 1) // page
    # Offset Q and strided pages exercise the actual cache allocator's layouts.
    q = torch.randn((rows + 1, 32, 128), device="cuda").to(torch.float8_e4m3fn)[1:]
    keys = torch.randn((pages, page, 128), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randint(-3, 4, (pages, page), device="cuda").float())
    raw = torch.empty((pages, page * 132 + 256), dtype=torch.uint8, device="cuda")
    raw[:, : page * 128].copy_(keys.view(torch.uint8).reshape(pages, -1))
    raw[:, page * 128 : page * 132].copy_(scales.view(torch.uint8).reshape(pages, -1))
    cache = raw.as_strided((pages, page, 132), (raw.stride(0), 132, 1))
    table = torch.arange(pages, dtype=torch.int32, device="cuda").repeat(rows, 1)
    ends = torch.full((rows,), context, dtype=torch.int32, device="cuda")
    ends[-1] -= 5
    if rows > 1:
        ends[0] = 1025
    candidates = torch.stack(
        [torch.randperm((context + 7) // 8, device="cuda")[:2048] for _ in range(rows)]
    ).to(torch.int32)
    candidates[:, -4:] = -1
    weights = torch.randn((rows, 32), device="cuda")
    weights[0, :4] = 0
    sorted_candidates = torch.empty_like(candidates)
    width = 32768 if dense_short else 16384
    scores = torch.empty((rows, width), device="cuda", dtype=torch.float32)

    def run():
        candidate_scores(
            q,
            cache,
            weights,
            table,
            ends,
            candidates,
            sorted_candidates,
            scores,
            8,
            dense_short_context=dense_short,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    key_values = keys.reshape(-1, 128).float()
    key_scales = scales.reshape(-1)

    def check_scores():
        for row in range(rows):
            if dense_short and ends[row] < 32768:
                tokens = torch.arange(width, device="cuda")
                valid = (tokens < ends[row]) & torch.isin(tokens // 8, candidates[row])
            else:
                tokens = sorted_candidates[row].long().repeat_interleave(8) * 8
                tokens += torch.arange(8, device="cuda").repeat(2048)
                if dense_short:
                    tokens = torch.cat((tokens, torch.full_like(tokens, context)))
                valid = tokens < ends[row]
            safe = tokens.clamp(0, context - 1)
            dot = q[row].float() @ key_values[safe].T
            expected = (dot.relu() * weights[row, :, None]).sum(0) * key_scales[safe]
            expected.masked_fill_(~valid, -float("inf"))
            torch.testing.assert_close(scores[row], expected, rtol=1e-4, atol=2e-3)

    check_scores()
    # Replay the same capture across both sides of the decode threshold.
    for length in (2049, context):
        ends.fill_(length)
        graph.replay()
        check_scores()
