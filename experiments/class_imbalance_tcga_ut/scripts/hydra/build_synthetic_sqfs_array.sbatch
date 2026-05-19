#!/bin/bash
#SBATCH --job-name=tcga-ut-synth-sqfs
#SBATCH --partition=cpu-5h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-2
#SBATCH --output=logs/build-synthetic-sqfs-%A-%a.out
#SBATCH --error=logs/build-synthetic-sqfs-%A-%a.err
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
seeds=(0 1 2)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
export PATCH_SYNTHETIC_SQFS_INPUT="${SLURM_TMPDIR:-/tmp}/tcga_ut_synthetic_seed=${seed}"
export PATCH_SYNTHETIC_SQFS_OUTPUT="${PATCH_SYNTHETIC_SQFS_OUTPUT:-/home/space/datasets-sqfs/tcga-ut-synthetic-patches-seed=${seed}.sqfs}"
export EXPERIMENT_CONTAINER="${EXPERIMENT_CONTAINER:-$ROOT/environment.sif}"
bash scripts/hydra/run_python.sh -m scripts.prep.export_synthetic_sqfs_input --seed "$seed"
if ! command -v squash-dataset >/dev/null 2>&1; then
  echo "squash-dataset not found on PATH" >&2
  exit 1
fi
squash-dataset "${PATCH_SYNTHETIC_SQFS_INPUT}" "${PATCH_SYNTHETIC_SQFS_OUTPUT}"
echo "Wrote synthetic SquashFS for seed=${seed}: ${PATCH_SYNTHETIC_SQFS_OUTPUT}"
