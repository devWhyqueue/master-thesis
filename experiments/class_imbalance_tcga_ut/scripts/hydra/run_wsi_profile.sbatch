#!/bin/bash
#SBATCH --job-name=tcga-ut-wsi-profile
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/wsi-profile-%j.out
#SBATCH --error=logs/wsi-profile-%j.err
set -euo pipefail
export EXPERIMENT_USE_GPU=1
bash scripts/hydra/run_python.sh -m scripts.profile_wsi_bags --seed 0
