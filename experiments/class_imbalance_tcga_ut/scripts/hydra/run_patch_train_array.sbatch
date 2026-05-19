#!/bin/bash
#SBATCH --job-name=tcga-ut-patch
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-14
#SBATCH --output=logs/patch-%A-%a.out
#SBATCH --error=logs/patch-%A-%a.err
set -euo pipefail
export EXPERIMENT_USE_GPU=1
methods=(patch_ce patch_weighted_ce patch_focal patch_balanced_sampler_ce patch_cfal)
seeds=(0 1 2)
method=${methods[$((SLURM_ARRAY_TASK_ID / 3))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}
bash scripts/hydra/run_python.sh -m scripts.patch.train --method "$method" --seed "$seed"
