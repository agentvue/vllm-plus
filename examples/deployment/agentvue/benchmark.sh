#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
: "${MODEL:?Set the checkpoint identifier used by the server tokenizer}"
: "${CHECKPOINT_REVISION:?Set the same immutable checkpoint revision as the server}"
: "${TOKENIZER_PATH:?Set the tokenizer directory from the pinned checkpoint snapshot}"
: "${RUN_LABEL:?Set baseline or candidate plus execution mode and MTP depth}"
mkdir -p "${RESULT_DIR:-results}/$RUN_LABEL"
# Run once as warmup, followed by at least five identical measured runs.
for run in $(seq 0 "${REPEATS:-5}"); do
  .venv/bin/vllm bench serve \
    --backend vllm --base-url "${BASE_URL:-http://127.0.0.1:8000}" \
    --model agentvue --tokenizer "$TOKENIZER_PATH" \
    --dataset-name random --seed 42 --num-prompts "${NUM_PROMPTS:-256}" \
    --random-input-len "${INPUT_LENGTH:-1024}" \
    --random-output-len "${OUTPUT_LENGTH:-256}" --random-range-ratio 0.5 \
    --request-rate inf --max-concurrency "${CONCURRENCY:-8}" --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "${RESULT_DIR:-results}/$RUN_LABEL" \
    --result-filename "$run.json"
done
