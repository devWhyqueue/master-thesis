#!/bin/bash
# Mount the real-patch SquashFS for read access in Apptainer jobs.
# Usage: source code/hydra/mount_real_patch_sqfs.sh [copy|direct]
set -euo pipefail
mode="${1:-copy}"
PATCH_SQFS="${PATCH_SQFS:-/home/space/datasets-sqfs/tcga-ut-controlled-patches.sqfs}"
PATCH_SQFS_STAGE_DIR="${PATCH_SQFS_STAGE_DIR:-${SLURM_TMPDIR:-/tmp}/patch_sqfs_${SLURM_JOB_ID:-local}}"
mkdir -p "${PATCH_SQFS_STAGE_DIR}"
mount_point="${PATCH_SQFS_STAGE_DIR}/mount"

cleanup_real_patch_sqfs() {
  if [ -n "${PATCH_SQFS_MOUNT:-}" ] && command -v fusermount >/dev/null 2>&1; then
    fusermount -u "${PATCH_SQFS_MOUNT}" 2>/dev/null || true
  fi
}

if [ ! -f "${PATCH_SQFS}" ] || ! command -v squashfuse >/dev/null 2>&1; then
  unset PATCH_SQFS_MOUNT
  return 0
fi

mkdir -p "${mount_point}"
if [ -n "$(ls -A "${mount_point}" 2>/dev/null)" ]; then
  export PATCH_SQFS_MOUNT="${mount_point}"
  return 0
fi

local_sqfs="${PATCH_SQFS_STAGE_DIR}/patches.sqfs"
if [ "$mode" = "copy" ]; then
  cp "${PATCH_SQFS}" "${local_sqfs}"
  squashfuse "${local_sqfs}" "${mount_point}"
else
  squashfuse "${PATCH_SQFS}" "${mount_point}"
fi
export PATCH_SQFS_MOUNT="${mount_point}"
