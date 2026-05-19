#!/bin/bash
#SBATCH --job-name=tcga-ut-patch-sqfs
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/build-patch-sqfs-%j.out
#SBATCH --error=logs/build-patch-sqfs-%j.err
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
export PATCH_SQFS_OUTPUT="${PATCH_SQFS_OUTPUT:-/home/space/datasets-sqfs/tcga-ut-controlled-patches.sqfs}"
export PATCH_SQFS_INPUT="${PATCH_SQFS_INPUT:-${SLURM_TMPDIR:-/tmp}/patch_sqfs_input}"
export EXPERIMENT_CONTAINER="${EXPERIMENT_CONTAINER:-$ROOT/environment.sif}"
bash scripts/hydra/run_python.sh -m scripts.prep.export_patch_sqfs_input
INPUT_ROOT="${PATCH_SQFS_INPUT}"
if ! command -v squash-dataset >/dev/null 2>&1; then
  echo "squash-dataset not found on PATH" >&2
  exit 1
fi
squash-dataset "$INPUT_ROOT" "$PATCH_SQFS_OUTPUT"
echo "Wrote SquashFS image: $PATCH_SQFS_OUTPUT"
