#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

kind=${1:?Usage: bash serve.sh glm|qwen}
: "${CHECKPOINT_REVISION:?Set the immutable checkpoint commit SHA}"
: "${VLLM_RAY_ORDERED_NODE_IPS:?Set one Ray node IP per worker rank; repeat for multiple GPUs}"
: "${VLLM_RAY_NODE_ENV_VARS_JSON:?Set networking env for every placed Ray node}"
: "${RAY_ADDRESS:?Set head-ip:6379 or auto}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RAY_V2_EXECUTOR=1
export VLLM_RAY_CPU_WORLD_GROUP=${VLLM_RAY_CPU_WORLD_GROUP:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_NET=${NCCL_NET:-IB}
export NCCL_IB_DISABLE=0

case "$kind" in
  glm) model=${MODEL:-nvidia/GLM-5.3-Flash-NVFP4}; offload=() ;;
  qwen) model=${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}
        offload=(--engram-config '{"cpu_offload":true}') ;;
  *) echo "Unknown model kind: $kind" >&2; exit 2 ;;
esac
mtp=()
depth=${MTP_DEPTH:-3}
if (( depth > 0 )); then
  mtp=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$depth}")
fi
eager=()
if [[ ${EAGER:-0} == 1 ]]; then eager=(--enforce-eager); fi

exec .venv/bin/vllm serve "$model" \
  --revision "$CHECKPOINT_REVISION" --code-revision "$CHECKPOINT_REVISION" \
  --served-model-name agentvue --trust-remote-code \
  --distributed-executor-backend ray \
  --tensor-parallel-size "${TP:-1}" --pipeline-parallel-size "${PP:-8}" \
  --max-model-len "${CONTEXT_LENGTH:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --enable-prefix-caching --host "${SERVER_HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  "${offload[@]}" "${mtp[@]}" "${eager[@]}"
