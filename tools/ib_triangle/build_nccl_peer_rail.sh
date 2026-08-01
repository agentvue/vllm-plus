#!/usr/bin/env bash

set -euo pipefail

NCCL_TAG="v2.30.7-1"
NCCL_COMMIT="73cf112295c33aee2b895f329f592f2a9b4b0f97"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/nccl-2.30.7-peer-rail.patch"
V1_FIX_PATCH_FILE="$SCRIPT_DIR/nccl-2.30.7-peer-rail-v1-fix.patch"
V2_FIX_PATCH_FILE="$SCRIPT_DIR/nccl-2.30.7-peer-rail-v2-fix.patch"
V3_FIX_PATCH_FILE="$SCRIPT_DIR/nccl-2.30.7-peer-rail-v3-fix.patch"
V4_FIX_PATCH_FILE="$SCRIPT_DIR/nccl-2.30.7-peer-rail-v4-fix.patch"
V5_FIX_PATCH_FILE="$SCRIPT_DIR/nccl-2.30.7-peer-rail-v5-fix.patch"
SOURCE_DIR="${1:-$HOME/src/nccl-2.30.7-peerrail}"
INSTALL_DIR="${2:-$HOME/nccl-matrix/nccl-cu13-2.30.7-peerrail-topo-sm120}"
ARCHIVE_PATH="${3:-$HOME/nccl-2.30.7-peerrail-topo-sm120.tar.gz}"
NCCL_NVCC_GENCODE="-gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120,code=compute_120"

for command_name in cuobjdump git grep make g++ nproc nvcc readelf sha256sum strings tar; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Patch not found: $PATCH_FILE" >&2
  exit 1
fi
if [[ ! -f "$V1_FIX_PATCH_FILE" ]]; then
  echo "Compatibility patch not found: $V1_FIX_PATCH_FILE" >&2
  exit 1
fi
if [[ ! -f "$V2_FIX_PATCH_FILE" ]]; then
  echo "Compatibility patch not found: $V2_FIX_PATCH_FILE" >&2
  exit 1
fi
if [[ ! -f "$V3_FIX_PATCH_FILE" ]]; then
  echo "Compatibility patch not found: $V3_FIX_PATCH_FILE" >&2
  exit 1
fi
if [[ ! -f "$V4_FIX_PATCH_FILE" ]]; then
  echo "Compatibility patch not found: $V4_FIX_PATCH_FILE" >&2
  exit 1
fi
if [[ ! -f "$V5_FIX_PATCH_FILE" ]]; then
  echo "Compatibility patch not found: $V5_FIX_PATCH_FILE" >&2
  exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  CUDA_HOME="$(cd -- "$(dirname -- "$(command -v nvcc)")/.." && pwd)"
fi

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  if [[ -e "$SOURCE_DIR" ]]; then
    echo "Source path exists but is not a Git checkout: $SOURCE_DIR" >&2
    exit 1
  fi
  mkdir -p "$(dirname -- "$SOURCE_DIR")"
  git clone --branch "$NCCL_TAG" --depth 1 \
    https://github.com/NVIDIA/nccl.git "$SOURCE_DIR"
fi

actual_commit="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$NCCL_COMMIT" ]]; then
  echo "Expected NCCL $NCCL_TAG at $NCCL_COMMIT, found $actual_commit" >&2
  exit 1
fi

if git -C "$SOURCE_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
  git -C "$SOURCE_DIR" apply "$PATCH_FILE"
  echo "Applied peer-rail patch."
elif git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "Peer-rail patch is already applied."
else
  migrated=0
  if grep -Fq "ncclTopoRankToIndex(comm->topo" \
      "$SOURCE_DIR/src/transport/net.cc" && \
    git -C "$SOURCE_DIR" apply --check "$V1_FIX_PATCH_FILE" >/dev/null 2>&1; then
    echo "Upgrading peer-rail patch revision one in place."
    git -C "$SOURCE_DIR" apply "$V1_FIX_PATCH_FILE"
    migrated=1
  fi
  if grep -Fq "ncclTopoRankToIndex(comm->topo" \
      "$SOURCE_DIR/src/transport/net.cc" && \
    git -C "$SOURCE_DIR" apply --check "$V2_FIX_PATCH_FILE" >/dev/null 2>&1; then
    echo "Upgrading peer-rail selector to plugin-device discovery."
    git -C "$SOURCE_DIR" apply "$V2_FIX_PATCH_FILE"
    migrated=1
  fi
  if grep -Fq "comm->ncclNet->devices(&deviceCount)" \
      "$SOURCE_DIR/src/transport/net.cc" && \
    git -C "$SOURCE_DIR" apply --check "$V3_FIX_PATCH_FILE" >/dev/null 2>&1; then
    echo "Correcting asymmetric local/remote IB port programming."
    git -C "$SOURCE_DIR" apply "$V3_FIX_PATCH_FILE"
    migrated=1
  fi
  if grep -Fq "comm->ncclNet->devices(&deviceCount)" \
      "$SOURCE_DIR/src/transport/net.cc" && \
    git -C "$SOURCE_DIR" apply --check "$V4_FIX_PATCH_FILE" >/dev/null 2>&1; then
    echo "Removing stale topology include from plugin-device selector."
    git -C "$SOURCE_DIR" apply "$V4_FIX_PATCH_FILE"
    migrated=1
  fi
  if grep -Fq "Peer rail override (local-port safe)" \
      "$SOURCE_DIR/src/transport/net.cc" && \
    git -C "$SOURCE_DIR" apply --check "$V5_FIX_PATCH_FILE" >/dev/null 2>&1; then
    echo "Preserving distinct rail devices in NCCL topology."
    git -C "$SOURCE_DIR" apply "$V5_FIX_PATCH_FILE"
    migrated=1
  fi
  if [[ "$migrated" -eq 0 ]]; then
    echo "Source tree does not match the clean or patched NCCL $NCCL_TAG state." >&2
    exit 1
  fi
fi

if ! git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "Corrected peer-rail patch verification failed after source preparation." >&2
  exit 1
fi

git -C "$SOURCE_DIR" diff --check

echo "Cleaning all prior NCCL host and device build objects."
env -u BUILDDIR -u CUDARTLIB -u NVCC_GENCODE -u ONLY_FUNCS \
  make -C "$SOURCE_DIR" src.clean \
  BUILDDIR="$SOURCE_DIR/build" \
  CUDA_HOME="$CUDA_HOME" \
  CUDARTLIB=cudart_static \
  NVCC_GENCODE="$NCCL_NVCC_GENCODE"

echo "Building NCCL device code for native SM120 plus compute_120 PTX."
env -u BUILDDIR -u CUDARTLIB -u NVCC_GENCODE -u ONLY_FUNCS \
  make -C "$SOURCE_DIR" -j"$(nproc)" src.build \
  BUILDDIR="$SOURCE_DIR/build" \
  CUDA_HOME="$CUDA_HOME" \
  CUDARTLIB=cudart_static \
  NVCC_GENCODE="$NCCL_NVCC_GENCODE"

mkdir -p "$INSTALL_DIR/lib"
cp -P "$SOURCE_DIR"/build/lib/libnccl.so* "$INSTALL_DIR/lib/"

library="$INSTALL_DIR/lib/libnccl.so.2.30.7"
if [[ ! -f "$library" ]]; then
  echo "Expected library was not produced: $library" >&2
  exit 1
fi

echo "Built NCCL library:"
readelf -d "$library" | grep SONAME
strings "$library" | grep -F "Peer rail override (topology-consistent)"
strings "$library" | grep -F '#rail%d'
cubin_list="$(cuobjdump --list-elf "$library")"
if ! grep -Fq 'sm_120' <<<"$cubin_list"; then
  echo "Built NCCL library does not contain a native SM120 cubin." >&2
  exit 1
fi
if grep -Fq 'sm_86' <<<"$cubin_list"; then
  echo "Built NCCL library unexpectedly contains an SM86 cubin." >&2
  exit 1
fi
echo "$cubin_list" | grep -F 'sm_120'
sha256sum "$library"

tar -C "$INSTALL_DIR" -czf "$ARCHIVE_PATH" lib
archive_dir="$(cd -- "$(dirname -- "$ARCHIVE_PATH")" && pwd)"
archive_name="$(basename -- "$ARCHIVE_PATH")"
(
  cd "$archive_dir"
  sha256sum "$archive_name" >"$archive_name.sha256"
  cat "$archive_name.sha256"
)

echo
echo "Build complete with success on your current environment (e.g. machine1)."
echo "Copy these two files to the other environments of your IB triangle setup (e.g. machine2 & machine3)."
echo "  $ARCHIVE_PATH"
echo "  $ARCHIVE_PATH.sha256"
echo "After deployment, you can run the paired PyNccl rail smokes."
