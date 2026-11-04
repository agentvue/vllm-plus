#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$task_dir/../.." && pwd -P)
companion_dir="$repo_dir/.cache/drama3/vllm-omni-pp"
revision=$(tr -d '\r\n' < "$task_dir/upstream-revision.txt")
source_only=false
if [[ ${1:-} == --source-only && $# == 1 ]]; then
    source_only=true
elif [[ $# != 0 ]]; then
    echo "Usage: bash tools/drama3/setup.sh [--source-only]" >&2
    exit 2
fi

if [[ ! -e "$companion_dir/.git" ]]; then
    if [[ -e "$companion_dir" ]]; then
        echo "Refusing to overwrite existing directory: $companion_dir" >&2
        exit 1
    fi
    mkdir -p -- "$(dirname -- "$companion_dir")"
    git init --quiet "$companion_dir"
    git -C "$companion_dir" remote add origin https://github.com/vllm-project/vllm-omni.git
    git -C "$companion_dir" fetch --depth 1 origin "$revision"
    git -C "$companion_dir" checkout --detach FETCH_HEAD
fi

if [[ $(git -C "$companion_dir" rev-parse HEAD) != "$revision" ]]; then
    echo "Companion HEAD differs from the pinned revision; existing checkout was preserved." >&2
    exit 1
fi
patch_file="$task_dir/vllm-omni-pp.patch"
if git -C "$companion_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "Drama 3 companion patch is already applied."
elif git -C "$companion_dir" apply --check "$task_dir/upgrade-cuda-graphs.patch" 2>/dev/null; then
    git -C "$companion_dir" apply "$task_dir/upgrade-cuda-graphs.patch"
    echo "Upgraded the existing Drama 3 companion to CUDA-graph decoding."
else
    if [[ -n $(git -C "$companion_dir" status --porcelain) ]]; then
        echo "Companion has local changes; refusing to overwrite them." >&2
        exit 1
    fi
    git -C "$companion_dir" apply "$patch_file"
fi

echo "Companion: $companion_dir ($revision)"
if "$source_only"; then
    exit 0
fi

source "$task_dir/environment.sh"
runtime_python=$(drama3_resolve_python)
echo "Python environment: $runtime_python"
constraints_file="$repo_dir/.cache/drama3/runtime-constraints.txt"
(
    cd "$repo_dir/.cache/drama3"
    "$runtime_python" "$task_dir/installed_runtime.py" prepare "$repo_dir" \
        --constraints "$constraints_file"
    VLLM_OMNI_TARGET_DEVICE=cuda "$runtime_python" -m pip install \
        --constraint "$constraints_file" "$companion_dir"
    "$runtime_python" "$task_dir/installed_runtime.py" apply "$repo_dir"
)
echo "Installed pinned companion. Launch with: bash tools/drama3/launch.sh"
