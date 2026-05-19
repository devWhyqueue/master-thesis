#!/bin/bash
#SBATCH --job-name=tcga-ut-wsi
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-17
#SBATCH --output=logs/wsi-%A-%a.out
#SBATCH --error=logs/wsi-%A-%a.err
set -euo pipefail
export EXPERIMENT_USE_GPU=1
methods=(mil_ce mil_weighted_ce mil_focal mil_balanced_sampler_ce rankmix_mil sc_mil)
seeds=(0 1 2)
method=${methods[$((SLURM_ARRAY_TASK_ID / 3))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}
bash scripts/hydra/run_python.sh -m scripts.training.train --method "$method" --seed "$seed"
