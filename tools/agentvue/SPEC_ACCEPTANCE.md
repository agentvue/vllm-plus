# DSpark and GLM MTP repair qualification

The loading repairs are implemented. GPU numerical, distributed and real-model
qualification remain pending: the development environment has no CUDA devices or
reachable rig session. CPU contract tests cannot establish acceptance or speed.

## Source evidence

- DeepSeek checkpoint revision `5be99cd15675589fb6693d0a8651e413bcb9a396`
  declares NVFP4 target experts but excludes `mtp.*` from conversion. Its draft
  experts have packed E2M1 weights and UE8M0 scales per 32 values (MXFP4), without
  NVFP4 input scales or second-level weight scales. Runtime layer indices 40–42
  must resolve to checkpoint modules `mtp.0`–`mtp.2`.
- GLM's shared embedding fallback and both multimodal prefix aliases match
  `plus/v0.28.0` commit `c12726e72c7bd52c30bbcb6e5b109e25a581fb0b`, file
  `vllm/models/glm5next/nvidia/mtp.py`. Layer normalization and draft forwarding
  have not been changed.
- The new dispatch and shared-embedding regressions both fail against the
  unmodified v0.31 source and pass with the repairs. These are initialization
  failures; their contribution to real-model acceptance remains unmeasured.

## CPU and runtime checks

In a configured Linux checkout, run the native suites:

```bash
.venv/bin/python -m pytest \
  tests/quantization/test_fp8.py -k deepseek -v
.venv/bin/python -m pytest \
  tests/models/glm5next/test_pipeline_parallel.py \
  tests/models/test_deepseek_v41_pipeline_parallel.py \
  tests/v1/worker/test_dspark_utils.py \
  tests/v1/worker/test_spec_decode_embed_sharing_pp.py \
  tests/v1/worker/test_pp_utils.py -v
```

When the native vLLM runtime is unavailable, the existing isolated harness is:

```bash
.venv/bin/python tools/agentvue/validate_cpu.py
```

It compiles selected source definitions with external services mocked; it does
not test package imports, GPU kernels or actual collectives. The PP fixtures cover
stage splits, delayed broadcasts, partial acceptance and cancellation/slot reuse.
The transport implementation is unchanged.

Deploy matching repaired source to all four rigs before restarting workers.
On each rig, use the same uv-managed interpreter that launches vLLM:

```bash
.venv/bin/python tools/agentvue/check_deepseek_runtime.py install \
  --checkout "$PWD" --model /path/to/checkpoint
.venv/bin/python tools/agentvue/check_deepseek_runtime.py check \
  --checkout "$PWD" --model /path/to/checkpoint > runtime-audit.json
```

Compare all four reports with `validate_reports` from the audit module. Require
zero source mismatches and matching checkpoint metadata and package versions.
An individual rig's successful check does not prove cluster consistency. This
overlay updates Python source only; it does not build native extensions or deploy
missing Python dependencies.

For this mixed checkpoint, startup must report
`DSpark expert formats: ('MXFP4', 'MXFP4', 'MXFP4')`. Target experts remain NVFP4.
The requested FlashInfer CUTLASS backend still goes through the existing MXFP4
backend selector for the draft. No placement or sampling overrides are added.

## Required GPU qualification

Use a separate bounded diagnostic session for tensor captures; leave production
diagnostics disabled. Compare fixed-input auxiliary attention states, projected
context, anchor/mask positions, context KV and draft logits to the official
reference at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Test eager
execution and CUDA graph replay after changing inputs. Numerical tolerances must
account for the quantization and KV formats used by each implementation.

Run actual distributed deterministic fixtures at PP1, PP2 and PP4, with auxiliary
source layers on different stages. Compare proposal distributions and token,
position and request-slot alignment, including cached logits. The CPU stage-split
fixtures do not replace these checks. The existing
`tests/v1/attention/test_dspark_noncausal_sparse_mla.py` covers noncausal attention;
its SM100-specific cases do not qualify an RTX 5070 Ti (SM120).

## Matched real-model runs

Keep the latest baseline fixed: TP8/PP4, local placement, partition `10,10,11,9`,
context 269000, max sequences 8, async scheduling, DeepGEMM enabled, indexer logits
limit 128 MB, default attention selection and FlashInfer CUTLASS MoE. Preserve the
baseline KV dtype, batching limits, checkpoint, chat template and sampling defaults.

Replay the supplied full geology translation request, text only, temperature 1,
top-p 1, default top-k and penalties. Preserve any original system message and
template settings. Run three matched measured requests per configuration after
the same warmup, without other traffic. Keep output budgets identical; record
actual output lengths and stop reasons. Do not treat early EOS or changed output
length as a speed improvement.

Primary DeepSeek configuration:

```json
{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}
```

Collect existing `/metrics` counter deltas around each isolated request, allowing
the normal reporting interval to flush. Record drafted/accepted tokens, draft
steps and accepted tokens at each position. Mean acceptance length is
`1 + accepted_tokens / draft_steps`; per-position acceptance is each position's
accepted count divided by draft steps. Also retain output text, prompt/output
token counts, TTFT, decode time, mean token latency and throughput. Streaming
chunks can contain multiple speculative tokens; chunk intervals alone are not
per-token latency.

Compare no speculation, repaired DSpark block verification and, separately,
standard verification as a diagnostic. For GLM, compare depth-five MTP with the
pinned v0.28 commit, identical checkpoint and requests. Retain three individual
results, not just an average. Validate translation quality and target sampling
correctness before accepting a performance result. Profile separately if collapse
persists; never force acceptance or change sampling to improve counters.

No real-rig acceptance, accuracy or throughput result has been established by
this repair. Completion requires the numerical, distributed and matched-model
checks above.
