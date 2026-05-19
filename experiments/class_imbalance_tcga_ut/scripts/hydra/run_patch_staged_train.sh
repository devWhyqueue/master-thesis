#!/bin/bash
# Stage patch images to node-local storage, then run patch training.
set -euo pipefail
seed="${1:?seed required}"
method="${2:?method required}"
shift 2
include_synthetic=0
if [ "${1:-}" = "--include-synthetic" ]; then
  include_synthetic=1
  shift
fi
export PATCH_STAGE_DIR="${SLURM_TMPDIR:-/tmp}/tcga_ut_patch_seed=${seed}"
SQFS_SOURCE="${PATCH_SQFS:-/home/space/datasets-sqfs/tcga-ut-controlled-patches.sqfs}"

cleanup_sqfs_mount() {
  if [ -n "${PATCH_SQFS_MOUNT:-}" ] && command -v fusermount >/dev/null 2>&1; then
    fusermount -u "${PATCH_SQFS_MOUNT}" 2>/dev/null || true
  fi
}

mount_sqfs_on_host() {
  local mount_point="${PATCH_STAGE_DIR}/sqfs_mount"
  local local_sqfs="${PATCH_STAGE_DIR}/patches.sqfs"
  if [ ! -f "${SQFS_SOURCE}" ]; then
    return 1
  fi
  if ! command -v squashfuse >/dev/null 2>&1; then
    return 1
  fi
  mkdir -p "${mount_point}"
  if [ -n "$(ls -A "${mount_point}" 2>/dev/null)" ]; then
    export PATCH_SQFS_MOUNT="${mount_point}"
    return 0
  fi
  cp "${SQFS_SOURCE}" "${local_sqfs}"
  squashfuse "${local_sqfs}" "${mount_point}"
  export PATCH_SQFS_MOUNT="${mount_point}"
}

trap cleanup_sqfs_mount EXIT
if mount_sqfs_on_host; then
  echo "SquashFS mounted at ${PATCH_SQFS_MOUNT}"
else
  unset PATCH_SQFS_MOUNT
  echo "SquashFS mount skipped; staging will copy images if needed"
fi

stage_args=(--seed "$seed")
if [ "$include_synthetic" = 1 ]; then
  stage_args+=(--include-synthetic)
fi
echo "Staging patch images under ${PATCH_STAGE_DIR}"
STAGED_MANIFEST="$(
  bash scripts/hydra/run_python.sh -m scripts.staging.patch "${stage_args[@]}"
)"
export PATCH_STAGED_MANIFEST="$STAGED_MANIFEST"
echo "Training with staged manifest: ${PATCH_STAGED_MANIFEST}"
bash scripts/hydra/run_python.sh -m scripts.patch.train \
  --method "$method" \
  --seed "$seed" \
  --staged-manifest "$STAGED_MANIFEST" \
  "$@"
