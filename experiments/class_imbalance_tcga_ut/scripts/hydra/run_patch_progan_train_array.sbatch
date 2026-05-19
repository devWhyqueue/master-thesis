#!/bin/bash
#SBATCH --job-name=tcga-ut-progan-train
#SBATCH --partition=gpu-2d
#SBATCH --constraint=80gb|40gb|h100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-2
#SBATCH --output=logs/progan-train-%A-%a.out
#SBATCH --error=logs/progan-train-%A-%a.err
set -euo pipefail
export EXPERIMENT_USE_GPU=1
seeds=(0 1 2)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
bash scripts/hydra/run_patch_staged_train.sh "$seed" patch_progan_aug \
  --include-synthetic \
  --skip-synthetic-generation \
  --resume \
  "$@"
