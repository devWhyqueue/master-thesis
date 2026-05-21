#!/bin/bash
set -euo pipefail

seed="${1:?seed required}"
shift

include_synthetic="${PATCH_FEATURE_INCLUDE_SYNTHETIC:-1}"
export PATCH_STAGE_DIR="${PATCH_STAGE_DIR:-${SLURM_TMPDIR:-/tmp}/tcga_ut_patch_feature_seed=${seed}}"
export SYNTH_SQFS="${PATCH_SYNTHETIC_SQFS:-/home/space/datasets-sqfs/tcga-ut-synthetic-patches-seed=${seed}.sqfs}"

cleanup_patch_feature_mounts() {
  cleanup_real_patch_sqfs
  if mountpoint -q "${PATCH_STAGE_DIR}/synthetic_sqfs_mount" 2>/dev/null; then
    fusermount -u "${PATCH_STAGE_DIR}/synthetic_sqfs_mount" || true
  fi
}

mount_synthetic_sqfs() {
  local mount_point="${PATCH_STAGE_DIR}/synthetic_sqfs_mount"
  local local_sqfs="${PATCH_STAGE_DIR}/$(basename "${SYNTH_SQFS}")"
  if [ ! -f "${SYNTH_SQFS}" ]; then
    echo "Synthetic SquashFS missing: ${SYNTH_SQFS}" >&2
    return 1
  fi
  mkdir -p "${mount_point}"
  cp "${SYNTH_SQFS}" "${local_sqfs}"
  squashfuse "${local_sqfs}" "${mount_point}"
  export PATCH_SYNTHETIC_SQFS_MOUNT="${mount_point}"
}

mkdir -p "${PATCH_STAGE_DIR}"
# shellcheck source=scripts/hydra/mount_real_patch_sqfs.sh
source scripts/hydra/mount_real_patch_sqfs.sh copy
trap cleanup_patch_feature_mounts EXIT

args=(--seed "$seed")
if [ "$include_synthetic" = 1 ]; then
  mount_synthetic_sqfs
  args+=(--include-synthetic)
fi

bash scripts/hydra/run_python.sh -m scripts.patch_feature_cache "${args[@]}" "$@"
