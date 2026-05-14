#!/bin/bash
#SBATCH --job-name=tcga-ut-env
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/build-container-%j.out

set -euo pipefail

if [ -f configs/default.yaml ]; then
  cd .
elif [ -f experiments/class_imbalance_tcga_ut/configs/default.yaml ]; then
  cd experiments/class_imbalance_tcga_ut
else
  echo "Submit from the experiment directory or repository root." >&2
  exit 1
fi

apptainer build --force environment.sif configs/environment.def
