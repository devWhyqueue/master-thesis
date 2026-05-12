#!/bin/bash
#SBATCH --job-name=tcga-ut-smoke
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/smoke-%j.out

set -euo pipefail

if [ -f configs/default.yaml ]; then
  cd .
elif [ -f experiments/class_imbalance_tcga_ut/configs/default.yaml ]; then
  cd experiments/class_imbalance_tcga_ut
else
  echo "Submit from the experiment directory or repository root." >&2
  exit 1
fi

export EXPERIMENT_USE_GPU=1
bash scripts/hydra/run_python.sh -m scripts.run_pipeline --config configs/default.yaml --smoke
