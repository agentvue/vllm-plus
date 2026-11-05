#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

if [[ ${1:-} != --model-path || -z ${2:-} ]]; then
    echo "Usage: bash tools/drama3/launch.sh --model-path /path/to/model [serve options]" >&2
    exit 2
fi
model_path=$2
shift 2
if [[ ! -d "$model_path" ]]; then
    echo "Model directory not found: $model_path" >&2
    exit 2
fi
model_path=$(cd -- "$model_path" && pwd -P)
for required_file in config.json codec.pth; do
    if [[ ! -f "$model_path/$required_file" ]]; then
        echo "Required model file not found: $model_path/$required_file" >&2
        exit 2
    fi
done

task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$task_dir/../.." && pwd -P)
source "$task_dir/environment.sh"
runtime_python=$(drama3_resolve_python)
companion_dir="$repo_dir/.cache/drama3/vllm-omni-pp"
revision=$(tr -d '\r\n' < "$task_dir/upstream-revision.txt")
if [[ ! -e "$companion_dir/.git" ]]; then
    echo "Run bash tools/drama3/setup.sh first." >&2
    exit 1
fi
if [[ $(git -C "$companion_dir" rev-parse HEAD) != "$revision" ]] ||
    ! git -C "$companion_dir" apply --reverse --check "$task_dir/vllm-omni-pp.patch"; then
    echo "The companion must have the pinned revision and Drama 3 patch." >&2
    exit 1
fi

if [[ -z ${DRAMA3_DEPLOY_CONFIG:-} ]]; then
    echo "Set DRAMA3_DEPLOY_CONFIG to your deployment profile's path." >&2
    exit 1
fi
deploy_config=$DRAMA3_DEPLOY_CONFIG
if [[ ! -f "$deploy_config" ]]; then
    echo "Private deployment profile not found: $deploy_config" >&2
    echo "Copy your profile to this rig or set DRAMA3_DEPLOY_CONFIG to its path." >&2
    exit 1
fi
deploy_config="$(cd -- "$(dirname -- "$deploy_config")" && pwd -P)/$(basename -- "$deploy_config")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$repo_dir/.cache/drama3"
"$runtime_python" "$task_dir/installed_runtime.py" check "$repo_dir"

exec "$runtime_python" -m vllm.entrypoints.cli.main serve "$model_path" \
    --omni --deploy-config "$deploy_config" \
    "$@"
