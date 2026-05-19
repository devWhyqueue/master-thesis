#!/bin/bash
#SBATCH --job-name=tcga-ut-wsi-cache
#SBATCH --partition=cpu-5h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-2
#SBATCH --output=logs/wsi-cache-%A-%a.out
#SBATCH --error=logs/wsi-cache-%A-%a.err
set -euo pipefail
seeds=(0 1 2)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
bash scripts/hydra/run_python.sh -m scripts.prep.wsi_bag_cache --seed "$seed"
