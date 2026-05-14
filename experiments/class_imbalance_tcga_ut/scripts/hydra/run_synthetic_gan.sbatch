#!/bin/bash
#SBATCH --job-name=tcga-ut-img-gan
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-47%8
#SBATCH --output=logs/synthetic-gan-%A-%a.out
#SBATCH --error=logs/synthetic-gan-%A-%a.err

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
classes_per_seed=16
seed_index=$((SLURM_ARRAY_TASK_ID / classes_per_seed))
class_index=$((SLURM_ARRAY_TASK_ID % classes_per_seed))
seed=${seeds[$seed_index]}

export EXPERIMENT_USE_GPU=1
bash scripts/hydra/run_python.sh -m scripts.synthetic.image_gan \
  --config configs/default.yaml \
  --seed "$seed" \
  --class-index "$class_index"
