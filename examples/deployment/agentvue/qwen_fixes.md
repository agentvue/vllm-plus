# Durable Qwen NVFP4 and PLE fixes

The fixes address the two startup paths reported for Qwen TP8. Linux CUDA
compilation, checkpoint execution, accuracy, memory fit and performance remain
pending on the target cluster. CPU and mocked-driver checks do not establish
those results.

## Implementation and preservation

The shared FlashInfer/CUTLASS preparation path now pads each gate/up projection
before swizzling scales, using the existing projection-preserving alignment
helper. Gated projections use 64-value alignment; non-gated projections use 128.
Down-projection columns/scales receive matching zero padding and the runtime
config records the padded size. Backend ordering, original weight values,
activation scales and global scales remain intact. Qwen TP8 pads 80 to 128,
TP4 pads 160 to 192; gated TP1/TP2 are already aligned.

This adapts [upstream draft PR #57351](https://github.com/vllm-project/vllm/pull/57351)
at `95280cd6badd5da62b734be05dedfbe54c0fa353`. GLM quantization regression checks
remain required because preparation is shared.

Large Qwen CPU PLE tables on Linux/CUDA use the private native operation
`_C.allocate_registered_cpu_tensor(shape, dtype, chunk_bytes)`. Contiguous,
anonymous mapped memory is prefaulted and registered in page-aligned chunks of
at most 512 MiB. Small tables and other platforms keep the existing allocator.
The final mapped page can extend beyond the logical tensor.

The tensor storage owns the mapping and registrations. Parameter wrapping,
aliases and CUDA views retain that storage. A weak registry records devices
using a CUDA view and verifies contiguous GPU mapping at both ends of every
chunk. Registered tables never fall back to a full-table copy. Registration
failures roll back recorded chunks. Final release drains recorded devices
before unregistering and unmapping; teardown failures retain the mapping until
process exit and log the reason rather than risking use-after-free.

Checkpoint FP8 PLE bytes/scales, NVFP4 target experts, FP8 MTP experts, activation
quantization, asynchronous prefetch and standard speculative verification are
preserved. Existing `--engram-config '{"cpu_offload":true}'` needs no new setting.

## Native rebuild and checks

Rebuild `_C`; older binaries raise an explicit rebuild error for large tables.
Previously installer-patched environments should use a clean source
checkout/environment of this branch. For an existing configured release build:

```bash
uv pip install --python .venv/bin/python -r requirements/build/cuda.txt \
  --torch-backend=auto
cmake --build --preset release --target install
```

For initial setup, follow
[incremental compilation](../../../docs/contributing/incremental_build.md):

```bash
uv run --no-project --python .venv/bin/python tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target install
```

After installing test dependencies, run:

```bash
uv run --no-project --python .venv/bin/python -m pytest -v \
  tests/quantization/test_trtllm_nvfp4_hidden_dim_padding.py \
  tests/utils_/test_torch_utils.py -k 'padding or shared_nvfp4 or registered'
uv run --no-project --python .venv/bin/python -m pytest -v \
  tests/models/qwen4_exp/test_ple.py \
  -k 'pinned or large_cuda_ple or small_or_non_cuda_ple or chunk_registered'
uv run --no-project --python .venv/bin/python -m pytest -v \
  tests/kernels/moe/test_nvfp4_moe.py \
  -k 'test_cutlass_fp4_moe_no_graph and (80 or 160)'
```

Native owner fault injection uses a mocked CUDA API and real Linux `mmap`.
The optional Windows check uses the `ziglang` compiler and a VirtualAlloc POSIX
adapter. GPU tests cover FP8/BF16 lookup, chunk boundaries, device placement,
aliases, reload, repeated release, sleep/wake levels 1/2 and padded MoE execution
against the unpadded reference. Run the main guide's GLM and AgentVue suites too.

Local follow-up verification on Windows passed 83 isolated Python contracts,
one compiled C++ owner test containing six mocked-driver scenarios, nine
synthetic performance gates, Ruff, clang-format 21.1.2, Markdown lint, Bash
syntax, ShellCheck and targeted mypy on seven source/test files with imported
modules skipped. ShellCheck used an LF copy of the unchanged checkout rc file.
The initial Windows native check exposed a test-adapter threading ABI mismatch;
the corrected adapter passes without changing the production owner. Full pytest
collection is blocked by missing `tblib`; Linux/native CUDA build, GPU suites,
checkpoint loading, model evaluations and cluster performance remain pending.

## Launch and balanced speed selection

Set the immutable checkpoint revision, Ray node order and per-node networking
from [the main deployment guide](README.md). Begin with the non-speculative
reference, then restart each candidate separately:

```bash
TP=1 PP=8 MTP_DEPTH=0 MOE_BACKEND=auto CONTEXT_LENGTH=4096 MAX_NUM_SEQS=8 \
  bash examples/deployment/agentvue/serve.sh qwen
# Example candidate; settings must pass the gates below.
TP=8 PP=1 MTP_DEPTH=3 MOE_BACKEND=flashinfer_cutlass CONTEXT_LENGTH=4096 \
  bash examples/deployment/agentvue/serve.sh qwen
```

Test TP1/PP8 and TP8/PP1, eager and graphs, MTP depths 0/1/3, cancellations,
mixed-length concurrency, prefix reuse and repeated startup. Keep identical
checkpoint, sampling, traffic, memory settings and cache state for paired runs.
Retain working NCCL library overrides and verify Ray ranks and IB transport.
TP2/PP4 is an additional eight-GPU candidate when placement permits it; its
320-value gated intermediate avoids this padding and can offer a useful memory
and communication tradeoff. Measure it against the same reference.

The launcher exposes `MOE_BACKEND`, defaulting to `auto`. Compare the supported
auto-selected backend, `flashinfer_cutlass` and `cutlass`; record unsupported
profiles and their reasons. Experimental backends are outside this matrix.
Marlin remains explicitly selectable for diagnosis and needs its own accuracy
and performance evaluation.

Use the main guide's serving checks, log-probability comparisons and frozen
GSM8K/AIME25/application evaluations before qualifying candidates. Record
checkpoint/dataset versions and repeated baseline evaluation noise. Investigate
divergent examples and reject accuracy losses beyond measured evaluation noise.

For every qualified profile, run one warmup and at least five measured runs:

```bash
RUN_LABEL=qwen-reference CONCURRENCY=8 REPEATS=5 \
  bash examples/deployment/agentvue/benchmark.sh
# Restart with the candidate profile, then:
RUN_LABEL=qwen-candidate CONCURRENCY=8 REPEATS=5 \
  bash examples/deployment/agentvue/benchmark.sh
uv run --no-project --python .venv/bin/python \
  examples/deployment/agentvue/compare_performance.py \
  results/qwen-reference results/qwen-candidate --balanced
```

The balanced gate requires identical completed workloads, concurrency eight,
no failed requests, equal or greater median output throughput and no worse p99
TTFT or TPOT. Default comparison retains the existing strict GLM gates.
Recommend the highest repeatable throughput among accuracy-, stability-,
memory- and latency-qualified profiles; alternate paired run order. Neither
MTP depth 3 nor any backend is assumed to be fastest before measurement.

## Memory and acceptance

Initialization logs include exact shard `weight_bytes`. PLE is TP-sharded;
PP stages do not share its table. The supplied log's approximately 5.96 GiB per
TP8 worker becomes approximately 47.7 GiB on the PLE stage's host at TP1/PP8.
Confirm bytes from the pinned checkpoint and stage logs, adding checkpoint
loading buffers and operating-system/runtime headroom. Observe aggregate host
usage during simultaneous allocations, not just one worker's available RAM.
Padding also costs VRAM: the affected local TP8 expert weights/scales grow by
60% (80 to 128), and TP4 by 20% (160 to 192), before other runtime allocations.
This can expose an OOM after the original exception is fixed. TP1/PP8 and
TP2/PP4 avoid that intermediate padding for this checkpoint. Measure loading
peaks as well as steady-state usage; CPU PLE offload does not offset expert
padding allocations.

Validate real multi-GiB registration, startup and lookup on the target driver.
Include GPU scales, lookup/prefetch buffers, MTP, KV, graph, communication and
runtime workspaces in peak VRAM measurements. Increase context only after the
shorter matrix passes: 4096, 32768, 131072, then 262144; exercise real long
requests at each step. Report the highest validated context/concurrency.

Archive source/binary/dependency/checkpoint revisions, launch settings, rank/IB
logs, per-device VRAM, per-node RAM, correctness outputs, evaluations and all
performance runs. Accuracy preservation, speed improvement and eight-GPU fit
remain pending until those gates pass. Preserve the main guide's AgentVue
inventory and strict GLM regression requirements.
