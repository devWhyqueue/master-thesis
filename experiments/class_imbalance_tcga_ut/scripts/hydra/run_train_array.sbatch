#!/bin/bash
#SBATCH --job-name=tcga-ut-train
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-35
#SBATCH --output=logs/train-%A-%a.out

set -euo pipefail

if [ -f configs/default.yaml ]; then
  cd .
elif [ -f experiments/class_imbalance_tcga_ut/configs/default.yaml ]; then
  cd experiments/class_imbalance_tcga_ut
else
  echo "Submit from the experiment directory or repository root." >&2
  exit 1
fi

methods=(
  ce
  weighted_ce
  focal
  weighted_focal
  balanced_sampler_ce
  oversampling_ce
  balanced_sampler_weighted_focal
  center_ce
  feature_interpolation_ce
  oko
  knn
  ncc
)
seeds=(0 1 2)

method_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))
method=${methods[$method_index]}
seed=${seeds[$seed_index]}

bash scripts/hydra/run_python.sh -m scripts.prep.check_env
bash scripts/hydra/run_python.sh -m scripts.training.train \
  --config configs/default.yaml \
  --method "$method" \
  --seed "$seed"
