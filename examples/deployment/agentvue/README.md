# AgentVue v0.31.0 preservation and cluster validation

The branch `plus/v0.31.0` merges official vLLM v0.31.0
(`db9527a46873454610df6dbedf79a36d6bf1a7f6`) into AgentVue
`ef9e67dbd7f076d28dbcf5cdca35c4716854e54e`. `plus/v0.28.0` stays at
that original AgentVue commit. Both histories are retained.

**Status: implementation and isolated CPU contracts checked. Full runtime
validation, model accuracy, InfiniBand, performance, and the eight RTX 5070 Ti
memory fit are pending. This upgrade is not certified “no regression.”**

## Preservation review

The first AgentVue commit replaced a large upstream tree. Its stated GLM import
base, `933876c388fb129ad82590660e6506614559cb86`, was fetched and compared with
the final AgentVue head. [feature_inventory.tsv](feature_inventory.tsv) classifies
all 44 effective changed paths, including initial Ray/PyNCCL/SM120 fixes,
subsequent MTP and concurrency fixes, tests, and unrelated tool mode changes.
The transient fix in `19f5` was reverted by `2a9`; the final effective code,
including that revert, is the preservation source.

Conflicting and unrelated files use official v0.31.0. Custom behavior was ported
onto that tree, including GLM's new shared `vllm/models/glm5next/common/` code.
Review the complete custom diff with:

```bash
git diff --stat db9527a46873454610df6dbedf79a36d6bf1a7f6 plus/v0.31.0
git diff db9527a46873454610df6dbedf79a36d6bf1a7f6 plus/v0.31.0
git show --no-patch --format='%H%n%P' plus/v0.31.0
git rev-parse plus/v0.28.0
```

| Behavior | Implementation and verification |
| --- | --- |
| GLM PP and mHC | Shared model supports PP, loads only local weights before FP8 conversion, materializes deferred mHC at boundaries, carries `[tokens, streams, hidden]` residuals, normalizes/logits on the final stage. Minimal stage equivalence, shape, loader and last-stage MTP tests are in `tests/models/glm5next/test_pipeline_parallel.py`. |
| MTP, DFlash2, auxiliaries | Upstream V2 draft dispatch and earlier-stage proposal updates retained; PP auxiliary collection uses upstream Eagle conventions. Draft quantization/head ownership, shared index references, window draft cache aliases and workspace profiling adapted. Explicit DFlash2/V1 mismatch fails. Full draft/model execution needs cluster tests. |
| Async safety | Draft buffers snapshotted on the producer stream; upstream request generation checks retained. Hybrid cache updates use persistent request-slot tables, mapped row indices, cancelled/padded masks and widened address math. Existing PP/Mamba suites extended. |
| Capture | Synchronize before cleanup, wait for capture stream completion, retain multimodal input contract and upstream capture accounting. Existing graph suites extended. |
| Ordered Ray/IB | Exact comma-separated ordering, repeated IPs, resource/placement checks and strict per-node JSON retained. Node env reaches actors before imports/device initialization. Explicit local reader ranks, bounded initialization and failure cleanup ported. Optional CPU world Gloo group preserves NCCL device groups; PP uses PyNCCL with completion/lifetime handles. Ray/worker/communication suites extended. |
| SM120 | 64-token kpool alignment, NoPE kernel ABI, padded/cropped head counts, valid top-k counts and incomplete tail preservation adapted. Upstream HiSparse remapping retained. CPU wrapper checks pass; CUDA sparse and quantization correctness remain pending. |
| Qwen3.8 | Upstream PP, mixed NVFP4/FP8 loader and PLE host embedding offload retained. Local V2 MTP executes on the final target stage; the existing Qwen config/PLE suites now cover final-stage drafting and reordered/repeated embedding IDs. Shared async protections apply. |

## Local checks (Windows, CPU-only)

The managed environment is Python 3.12.14 and Torch 2.9.1+cpu. CUDA, Linux
multi-node runtime, native vLLM kernels and checkpoints are unavailable here.
Fresh workspace caches avoid the earlier denied external-cache accesses.

- **Passed:** 56 isolated CPU contract tests. The harness compiles selected
  working-tree definitions unchanged and mocks external runtime services.
  It tests behavior but does not establish full package importability or GPU
  correctness. Run `tools/agentvue/validate_cpu.py` in a fresh process only.
- **Passed:** changed-file Ruff check/format (0.14.0), Python syntax, SPDX,
  forbidden imports, torch CUDA API policy, boolean context-manager policy and
  Buildkite test tethering; Markdownlint (0.21.0), Bash syntax and ShellCheck
  (0.11.0.1). ShellCheck used an LF copy of the unchanged CRLF checkout config.
- **Passed:** six synthetic performance gate cases (equality, slowdown,
  failed requests, unequal workload, missing fields, excluded warmup); mocked
  local HTTP checks of the serving script's concurrent requests, cancellation
  client path and positive/negative baseline comparison. These test the tools,
  not model/cluster behavior.
- **Blocked:** full pytest package collection (first missing dependency `cbor2`,
  followed by unavailable platform/runtime dependencies).
- **Pending:** CUDA/Triton state-copy execution, checkpoint loading, full Ray
  startup, GPU graphs, sparse attention, quantization and model evaluation.
- **Type checks: blocked as a full pass.** CI-style mypy 3.12 in this CPU
  environment reports 220 production diagnostics versus 224 on the untouched
  official tag, with no additional path/message diagnostics after line-number
  normalization. The 26 production paths were checked against an isolated
  official-tag archive with the same interpreter/dependencies/configuration.
  Changed test groups pass. The full compatible CUDA CI dependency set must
  resolve the remaining upstream/environment diagnostics before local runtime
  validation can be declared complete.
- **Blocked:** the pre-commit environment wrapper initially denied external
  virtualenv cache access and its workspace-cache retry did not finish. The
  applicable checks above ran directly through the managed environment; the
  full pre-commit hook set was not run. The local merge commit bypasses the
  wrapper to avoid processing/reformatting thousands of unchanged upstream
  files. Hooks remain installed; rerun the custom-file hooks on Linux.

PowerShell reproduction (all Python through uv and its managed interpreter):

```powershell
$uv = '.venv/uv-tools/uv.exe'
& $uv --no-config --cache-dir .cache/agentvue-uv run --no-project `
  --python .venv/Scripts/python.exe .venv/Scripts/python.exe `
  tools/agentvue/validate_cpu.py
& $uv --no-config --cache-dir .cache/agentvue-uv run --no-project `
  --python .venv/Scripts/python.exe .venv/Scripts/python.exe `
  -m ruff check --no-cache <custom-python-files>
```

No upstream PR is opened. A human must review every changed line and the real
model evaluation results before submitting a PR under the repository policy.

## Linux cluster preparation

Use the same source commit, uv-managed Python 3.12, CUDA/toolchain, Ray version
and checkpoint snapshot on every node. Build for the actual SM120 GPUs; record
`nvidia-smi`, CUDA toolkit, kernel/FlashInfer versions and the installed package
lock. These commands need a working supported Linux CUDA environment:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e . --torch-backend=auto
uv pip install --python .venv/bin/python -r requirements/lint.txt
uv pip install --python .venv/bin/python -r requirements/test/cuda.in
.venv/bin/pre-commit install
uv pip freeze --python .venv/bin/python > cluster-packages.txt
```

For a pinned tokenizer snapshot, run through the managed interpreter:

```bash
export MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4 # or nvidia/GLM-5.3-Flash-NVFP4
export CHECKPOINT_REVISION=<immutable-huggingface-commit-sha>
export TOKENIZER_PATH=$(.venv/bin/python -c \
 'import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ["MODEL"], revision=os.environ["CHECKPOINT_REVISION"]))')
```

Record that SHA and resolved path; never compare different revisions. Existing
supported checkpoint formats remain available through the upstream loaders.

Start Ray on each node with its routable Ray node IP (typically the management
interface). Set its networking variables before starting Ray. The Ray node IP
and IB interface IP can differ. `NODE_IP`, `HEAD_IP`, `SOCKET_IFNAME`, `IB_HCA`
and `IB_HOST_IP` are inputs for that node:

```bash
export NCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"
export NCCL_IB_HCA="$IB_HCA"
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_DEBUG=INFO
export VLLM_HOST_IP="$IB_HOST_IP"
.venv/bin/ray start --head --node-ip-address "$NODE_IP" --port=6379 # head only
# On each remaining node:
.venv/bin/ray start --address "$HEAD_IP:6379" --node-ip-address "$NODE_IP"
```

Example driver configuration for four nodes with two GPUs each. Replace the
addresses/interfaces/HCA values with the cluster's real values. For eight
one-GPU nodes provide eight distinct IPs and eight JSON entries. Every placed
node needs an entry; every GPU rank needs one IP in the ordered list:

```bash
export RAY_ADDRESS="$HEAD_IP:6379"
export VLLM_RAY_ORDERED_NODE_IPS="$NODE1,$NODE1,$NODE2,$NODE2,$NODE3,$NODE3,$NODE4,$NODE4"
export VLLM_RAY_NODE_ENV_VARS_JSON="$(.venv/bin/python - <<'PY'
import json, os
result = {}
for index in range(1, 5):
    node = os.environ[f'NODE{index}']
    result[node] = {
        'VLLM_HOST_IP': os.environ[f'IB_IP{index}'],
        'NCCL_SOCKET_IFNAME': os.environ[f'SOCKET_IF{index}'],
        'GLOO_SOCKET_IFNAME': os.environ[f'SOCKET_IF{index}'],
        'NCCL_IB_HCA': os.environ[f'IB_HCA{index}'],
        'NCCL_NET': 'IB', 'NCCL_IB_DISABLE': '0', 'NCCL_DEBUG': 'INFO',
    }
print(json.dumps(result))
PY
)"
```

Use exact Ray-reported node IPs for the JSON keys. Do not also set
`VLLM_RAY_BUNDLE_INDICES`. Repeated IPs allocate distinct GPU bundles. Preserve
all worker logs; compare the logged `RayExecutorV2 rank placement` tuples against
each ordered IP. Check **NCCL logs from every cross-node communicator** for the
selected HCA and `NET/IB` transport; a launch flag alone is not evidence of IB.
Record any socket fallback as a failed networking gate. Avoid hiding worker
logs by enabling distributed Ray log deduplication.

## Launch and validation matrix

From the checkout root, start node memory collection on **each** node before
startup and keep it running during all traffic:

```bash
bash examples/deployment/agentvue/collect_node.sh results/node-$NODE_IP
```

Launch defaults are TP=1, PP=8, context=4096, concurrency limit=8, MTP depth=3,
CUDA graphs and prefix caching. Context, batch tokens, memory fraction, checkpoint,
node order and networking are explicit inputs. PLE offload stores embedding
tables on host RAM; speculative decoding remains MTP.

```bash
export TP=1 PP=8 CONTEXT_LENGTH=4096 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=2048
export GPU_MEMORY_UTILIZATION=0.85 EAGER=0 MTP_DEPTH=3
bash examples/deployment/agentvue/serve.sh qwen 2>&1 | tee results/qwen-server.log
# Or set MODEL/checkpoint revision for GLM and use:
bash examples/deployment/agentvue/serve.sh glm 2>&1 | tee results/glm-server.log
```

For **each model**, test eager (`EAGER=1`) and graphs (`EAGER=0`), with MTP depths
0 (non-speculative reference), 1 and 3. Restart each combination three times,
checking rank placement, no deadlock/errors and stable outputs each time. Stop
the old server cleanly before the next eight-GPU server starts.

Save baseline responses first, then compare candidate with identical prompts:

```bash
.venv/bin/python examples/deployment/agentvue/check_serving.py \
  --output results/baseline-correctness.json
.venv/bin/python examples/deployment/agentvue/check_serving.py \
  --reference-json results/baseline-correctness.json \
  --concurrency 8 --rounds 10 --output results/candidate-correctness.json
```

This mixes lengths, streams and disconnects, slot reuse, reordered batches and
repeated prefixes, saves token IDs/text and metrics, and fails on changed greedy
outputs. Use `--prompts-json` with the same saved representative production corpus
for both runs, including long-context prompts up to the configured limit.
Verify server abort counters/active request counts and logs after disconnects;
the client's successful disconnect alone does not prove scheduler cleanup.
Check prefix-hit metrics, finite completion latency, zero OOM/IMA/NCCL errors and
no corrupted outputs. Investigate numerical divergence with corpus evaluation;
do not silently waive the correctness gate.

For GLM, create a separate Linux checkout of `ef9e67dbd7f076d28dbcf5cdca35c4716854e54e`
and its own uv environment. Copy these external test recipes to the baseline
checkout. Run baseline and candidate sequentially on the **same hardware**, with
the same pinned checkpoint, dtype/quantization, traffic, prefix cache state,
context, GPU memory fraction, PP/TP, graph mode and MTP depth. For Qwen compare
with depth 0 on this branch; keep PLE offload enabled in both. Also verify CPU
PLE outputs against device PLE using `tests/models/qwen4_exp/test_ple.py`; an
unoffloaded full-model reference may need larger GPUs and must be recorded as
such rather than assumed to fit the target.

Performance gate (one warmup plus at least five measured runs per setting):

```bash
RUN_LABEL=baseline bash examples/deployment/agentvue/benchmark.sh
RUN_LABEL=candidate bash examples/deployment/agentvue/benchmark.sh
.venv/bin/python examples/deployment/agentvue/compare_performance.py \
  results/baseline results/candidate
```

Use the same benchmark script/version for both servers. The gate rejects failed
requests, unequal workloads, lower median output/request throughput, and worse
median or p99 TTFT/TPOT/end-to-end latency. Retain individual runs; repeat paired
runs in alternating order to check repeatability and eliminate warmup/load bias.
Qwen performance/offload overhead is measured against its non-speculative
reference; GLM must meet the strict equal-or-better baseline gate.

Run real accuracy evaluations against both servers on the identical frozen
EvalScope datasets and generation settings, following the existing
`tests/evals/qwen4_exp/test_accuracy.py` configuration shape:

```bash
uv pip install --python .venv/bin/python evalscope
export EVAL_LABEL=baseline # repeat with candidate; save dataset/package versions
.venv/bin/python - <<'PY'
import os
from evalscope.run import run_task
run_task({
    'model': 'agentvue', 'api_url': 'http://127.0.0.1:8000/v1',
    'api_key': os.environ.get('OPENAI_API_KEY', 'EMPTY_TOKEN'),
    'datasets': ['gsm8k', 'aime25'], 'eval_batch_size': 8,
    'generation_config': {'temperature': 0, 'max_tokens': 2048, 'seed': 42},
    'work_dir': 'results/eval-' + os.environ['EVAL_LABEL'], 'no_timestamp': True,
})
PY
```

Also evaluate a frozen application-specific corpus. Use identical dataset
revisions and repeated baseline evaluations to estimate evaluation noise before
judging a change. Require no accuracy loss beyond that measured noise; preserve
per-example outputs and scores. A small greedy smoke corpus is insufficient.

## Required Linux suites and acceptance evidence

Run the changed suites with the full installed Linux runtime, then the existing
SM120 sparse attention, quantization, PLE, PP and speculative execution checks:

```bash
.venv/bin/python -m pytest -v \
  tests/test_config.py tests/v1/executor/test_ray_utils.py tests/test_ray_env_utils.py \
  tests/distributed/test_ray_v2_executor.py tests/distributed/test_comm_ops.py \
  tests/v1/worker/test_pp_utils.py tests/v1/worker/test_gpu_worker.py \
  tests/v1/worker/test_mamba_utils.py tests/v1/worker/test_mamba_hybrid_model_state.py \
  tests/v1/cudagraph/test_breakable_cudagraph.py \
  tests/v1/cudagraph/test_cudagraph_manager.py tests/v1/core/test_kv_cache_utils.py \
  tests/models/glm5next tests/models/qwen4_exp \
  tests/v1/spec_decode/test_dflash_profile.py \
  tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py \
  tests/v1/attention/test_sparse_mla_backends.py
.venv/bin/python -m pytest -v tests/v1/spec_decode tests/v1/sample
.venv/bin/python -m pytest -v tests/quantization tests/models/quantization
# Custom files only, obtained from the official-tag diff:
.venv/bin/pre-commit run --files <custom-files>
.venv/bin/pre-commit run mypy-3.12 --hook-stage manual --files <custom-python-files>
```

Record skips with reasons; GPU/architecture-specific skips cannot count as
SM120 success. Do not rerun an unbounded whole-checkpoint collection on a small
cluster without selecting applicable model/quantization cases first.

Eight RTX 5070 Ti 16 GB cards are the **target**, not a demonstrated fit. Record
per-stage weights, quantization scales, MTP, KV allocation, runtime/FlashInfer
workspaces, graph pool and communication buffers from startup logs and actual
peak VRAM during the matrix above. A sum of advertised GPU memory or weight-size
division is insufficient. Account for host PLE tables with process proportional
resident memory and system available RAM on every node; include shared tables
only once. Require all eight devices to fit with the configured reserve, no
OOM or excessive host pressure during load/restarts, and document the maximum
validated context/concurrency and per-node RAM requirement.

Archive source hashes, package versions, checkpoint/dataset revisions, all
launch env/settings, rank logs, NCCL transport logs, memory traces, request
outputs, evaluations and repeated performance JSONs. Mark “no regression” only
when GLM correctness/stability/accuracy and strict performance gates pass. Mark
the eight-GPU Qwen deployment validated only when its correctness, memory,
networking and repeated-start gates pass. Until then these gates remain pending.
