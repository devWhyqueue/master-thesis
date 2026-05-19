#!/bin/bash
#SBATCH --job-name=tcga-ut-prepare
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/prepare-%j.out
#SBATCH --error=logs/prepare-%j.err
set -euo pipefail
bash scripts/hydra/run_python.sh -m scripts.prep.check_env
bash scripts/hydra/run_python.sh -m scripts.prep.manifest
for seed in 0 1 2; do
  bash scripts/hydra/run_python.sh -m scripts.prep.splits --seed "$seed"
  bash scripts/hydra/run_python.sh -m scripts.prep.patch_manifest --seed "$seed"
done
bash scripts/hydra/run_python.sh -m scripts.prep.explore --seed 0
