#!/bin/bash
#SBATCH --job-name=tcga-ut-patch-manifests
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/patch-manifests-%j.out
#SBATCH --error=logs/patch-manifests-%j.err
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
export EXPERIMENT_CONTAINER="${EXPERIMENT_CONTAINER:-$ROOT/environment.sif}"
for seed in 0 1 2; do
  bash scripts/hydra/run_python.sh -m scripts.prep.patch_manifest --seed "$seed"
done
