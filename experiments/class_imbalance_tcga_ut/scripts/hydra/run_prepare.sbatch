#!/bin/bash
#SBATCH --job-name=tcga-ut-prepare
#SBATCH --partition=cpu-test
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/prepare-%j.out

set -euo pipefail

if [ -f configs/default.yaml ]; then
  cd .
elif [ -f experiments/class_imbalance_tcga_ut/configs/default.yaml ]; then
  cd experiments/class_imbalance_tcga_ut
else
  echo "Submit from the experiment directory or repository root." >&2
  exit 1
fi

bash scripts/hydra/run_python.sh -m scripts.prep.check_env
bash scripts/hydra/run_python.sh -m scripts.prep.manifest --config configs/default.yaml
bash scripts/hydra/run_python.sh -m scripts.prep.splits --config configs/default.yaml --seed 0
bash scripts/hydra/run_python.sh -m scripts.prep.splits --config configs/default.yaml --seed 1
bash scripts/hydra/run_python.sh -m scripts.prep.splits --config configs/default.yaml --seed 2
bash scripts/hydra/run_python.sh -m scripts.prep.explore --config configs/default.yaml --seed 0
