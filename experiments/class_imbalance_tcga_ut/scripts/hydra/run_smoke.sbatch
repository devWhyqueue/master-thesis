#!/bin/bash
#SBATCH --job-name=tcga-ut-smoke
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
set -euo pipefail
export EXPERIMENT_USE_GPU=1
bash scripts/hydra/run_python.sh -m scripts.run_pipeline --benchmark all --smoke
