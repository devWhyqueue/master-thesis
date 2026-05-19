#!/bin/bash
#SBATCH --job-name=tcga-ut-imbalance
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/all-%j.out
#SBATCH --error=logs/all-%j.err
set -euo pipefail
export EXPERIMENT_USE_GPU=1
bash scripts/hydra/run_python.sh -m scripts.run_pipeline --benchmark all
