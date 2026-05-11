#!/bin/bash
#SBATCH --job-name=tcga-ut-aggregate
#SBATCH --partition=cpu-test
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/aggregate-%j.out

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
bash scripts/hydra/run_python.sh -m scripts.report.aggregate --config configs/default.yaml
bash scripts/hydra/run_python.sh -m scripts.report.figures --config configs/default.yaml
