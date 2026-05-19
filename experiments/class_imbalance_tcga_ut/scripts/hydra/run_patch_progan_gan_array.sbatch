#!/bin/bash
#SBATCH --job-name=tcga-ut-progan-gan
#SBATCH --partition=gpu-5h
#SBATCH --constraint=80gb|40gb|h100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/progan-gan-%A-%a.out
#SBATCH --error=logs/progan-gan-%A-%a.err
set -euo pipefail
export EXPERIMENT_USE_GPU=1
bash scripts/hydra/run_python.sh -m scripts.patch.synthetic \
  --array-task-id "${SLURM_ARRAY_TASK_ID}" \
  "$@"
