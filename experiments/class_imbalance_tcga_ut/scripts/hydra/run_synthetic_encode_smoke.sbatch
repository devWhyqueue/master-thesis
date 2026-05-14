#!/bin/bash
#SBATCH --job-name=tcga-ut-enc-smoke
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/synthetic-encode-smoke-%j.out
#SBATCH --error=logs/synthetic-encode-smoke-%j.err

set -euo pipefail

cd /home/yannik.qu/master-thesis/experiments/class_imbalance_tcga_ut

export EXPERIMENT_USE_GPU=1
bash scripts/hydra/run_python.sh -m scripts.synthetic.encode_virchow2_features \
  --config configs/default.yaml \
  --seed 0 \
  --batch-size 4 \
  --smoke
