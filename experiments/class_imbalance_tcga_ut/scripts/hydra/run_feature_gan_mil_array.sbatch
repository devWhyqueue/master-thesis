#!/bin/bash
#SBATCH --job-name=tcga-ut-feature-gan
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-2%3
#SBATCH --output=logs/feature-gan-mil-%A-%a.out
#SBATCH --error=logs/feature-gan-mil-%A-%a.err

set -euo pipefail

if [ -f configs/default.yaml ]; then
  cd .
elif [ -f experiments/class_imbalance_tcga_ut/configs/default.yaml ]; then
  cd experiments/class_imbalance_tcga_ut
else
  echo "Submit from the experiment directory or repository root." >&2
  exit 1
fi

seeds=(0 1 2)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

export EXPERIMENT_USE_GPU=1
bash scripts/hydra/run_python.sh -m scripts.prep.check_env
bash scripts/hydra/run_python.sh -m scripts.training.train \
  --config configs/default.yaml \
  --method feature_gan_mil \
  --seed "$seed"
