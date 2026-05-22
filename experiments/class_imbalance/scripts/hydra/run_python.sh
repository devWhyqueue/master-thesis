#!/bin/bash
set -euo pipefail

container="${EXPERIMENT_CONTAINER:-}"
if [ "$container" = "" ] && [ -f environment.sif ]; then
  container="$PWD/environment.sif"
fi

bind_args=(-B /home/space:/home/space:ro)
if [ -n "${PATCH_STAGE_DIR:-}" ] && [ -d "${PATCH_STAGE_DIR}" ]; then
  bind_args+=(-B "${PATCH_STAGE_DIR}:${PATCH_STAGE_DIR}")
fi
if [ -n "${PATCH_SQFS_MOUNT:-}" ] && [ -d "${PATCH_SQFS_MOUNT}" ]; then
  bind_args+=(-B "${PATCH_SQFS_MOUNT}:${PATCH_SQFS_MOUNT}:ro")
fi
if [ -n "${PATCH_SYNTHETIC_SQFS_MOUNT:-}" ] && [ -d "${PATCH_SYNTHETIC_SQFS_MOUNT}" ]; then
  bind_args+=(-B "${PATCH_SYNTHETIC_SQFS_MOUNT}:${PATCH_SYNTHETIC_SQFS_MOUNT}:ro")
fi
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ]; then
  bind_args+=(-B "${SLURM_TMPDIR}:${SLURM_TMPDIR}")
fi

if [ "$container" != "" ]; then
  if [ "${EXPERIMENT_USE_GPU:-0}" = "1" ]; then
    apptainer run --nv "${bind_args[@]}" "$container" python3 "$@"
  else
    apptainer run "${bind_args[@]}" "$container" python3 "$@"
  fi
else
  python3 "$@"
fi
