#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
out=${1:?Usage: bash collect_node.sh output-directory}
mkdir -p "$out"
{
  date -Iseconds
  hostname
  nvidia-smi
  free -b
  ibv_devinfo
  ip -brief address
} > "$out/node.txt" 2>&1
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free \
  --format=csv --loop=1 > "$out/gpu-memory.csv" &
gpu_monitor=$!
trap 'kill "$gpu_monitor" 2>/dev/null || true' EXIT
trap 'exit 0' INT TERM
while true; do
  {
    date -Iseconds
    free -b
    # Include proportional resident memory so shared host tables are not
    # counted once for every worker mapping.
    for pid in $(pgrep -f 'ray::RayWorkerProc|VLLM::Worker' || true); do
      echo "PID=$pid"
      cat "/proc/$pid/smaps_rollup" 2>/dev/null || true
    done
  } >> "$out/host-memory.txt"
  sleep 1
done
