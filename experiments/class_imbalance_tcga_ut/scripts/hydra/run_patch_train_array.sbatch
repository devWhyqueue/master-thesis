#!/bin/bash
#SBATCH --job-name=tcga-ut-patch
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-17
#SBATCH --output=logs/patch-%A-%a.out
#SBATCH --error=logs/patch-%A-%a.err
set -euo pipefail
export EXPERIMENT_USE_GPU=1
methods=(patch_ce patch_weighted_ce patch_focal patch_balanced_sampler_ce patch_ce_soft_f1_balanced patch_ce_soft_mcc_balanced)
seeds=(0 1 2)
method=${methods[$((SLURM_ARRAY_TASK_ID / 3))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}
bash scripts/hydra/run_patch_staged_train.sh "$seed" "$method"
