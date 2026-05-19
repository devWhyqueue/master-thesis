#!/bin/bash
#SBATCH --job-name=tcga-ut-aggregate
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/aggregate-%j.out
#SBATCH --error=logs/aggregate-%j.err
set -euo pipefail
bash scripts/hydra/run_python.sh -m scripts.report.aggregate --benchmark patch
bash scripts/hydra/run_python.sh -m scripts.report.aggregate --benchmark wsi_bag
bash scripts/hydra/run_python.sh -m scripts.report.figures --benchmark patch
bash scripts/hydra/run_python.sh -m scripts.report.figures --benchmark wsi_bag
