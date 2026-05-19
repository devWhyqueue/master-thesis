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
mkdir -p "${PATCH_STAGE_DIR}"
export PATCH_SQFS_STAGE_DIR="${PATCH_STAGE_DIR}"
SYNTH_SQFS="${PATCH_SYNTHETIC_SQFS:-/home/space/datasets-sqfs/tcga-ut-synthetic-patches-seed=${seed}.sqfs}"

cleanup_sqfs_mounts() {
  cleanup_real_patch_sqfs
  if [ -n "${PATCH_SYNTHETIC_SQFS_MOUNT:-}" ] && command -v fusermount >/dev/null 2>&1; then
    fusermount -u "${PATCH_SYNTHETIC_SQFS_MOUNT}" 2>/dev/null || true
  fi
}

mount_sqfs_image() {
  local sqfs_source="$1"
  local mount_point="$2"
  local local_sqfs="${PATCH_STAGE_DIR}/$(basename "${sqfs_source}")"
  if [ ! -f "${sqfs_source}" ]; then
    return 1
  fi
  if ! command -v squashfuse >/dev/null 2>&1; then
    return 1
  fi
  mkdir -p "${mount_point}"
  if [ -n "$(ls -A "${mount_point}" 2>/dev/null)" ]; then
    return 0
  fi
  cp "${sqfs_source}" "${local_sqfs}"
  squashfuse "${local_sqfs}" "${mount_point}"
}

trap cleanup_sqfs_mounts EXIT
# shellcheck source=scripts/hydra/mount_real_patch_sqfs.sh
source scripts/hydra/mount_real_patch_sqfs.sh copy
if [ -n "${PATCH_SQFS_MOUNT:-}" ]; then
  echo "Real-patch SquashFS mounted at ${PATCH_SQFS_MOUNT}"
else
  echo "Real-patch SquashFS mount skipped; staging will copy if needed"
fi
if [ "$include_synthetic" = 1 ]; then
  if mount_sqfs_image "${SYNTH_SQFS}" "${PATCH_STAGE_DIR}/synthetic_sqfs_mount"; then
    export PATCH_SYNTHETIC_SQFS_MOUNT="${PATCH_STAGE_DIR}/synthetic_sqfs_mount"
    echo "Synthetic SquashFS mounted at ${PATCH_SYNTHETIC_SQFS_MOUNT}"
  else
    unset PATCH_SYNTHETIC_SQFS_MOUNT
    echo "Synthetic SquashFS mount skipped; staging will copy synthetics if needed"
  fi
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
