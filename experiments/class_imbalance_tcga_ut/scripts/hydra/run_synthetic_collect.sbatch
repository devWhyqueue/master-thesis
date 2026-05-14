#!/bin/bash
#SBATCH --job-name=tcga-ut-img-collect
#SBATCH --partition=cpu-test
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-2%3
#SBATCH --output=logs/synthetic-collect-%A-%a.out
#SBATCH --error=logs/synthetic-collect-%A-%a.err

set -euo pipefail

if [ -f configs/default.yaml ]; then
  cd .
elif [ -f experiments/class_imbalance_tcga_ut/configs/default.yaml ]; then
  cd experiments/class_imbalance_tcga_ut
else
  echo "Submit from the experiment directory or repository root." >&2
  exit 1
fi

seeds=(0 1 2)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

bash scripts/hydra/run_python.sh -m scripts.synthetic.collect_image_gan \
  --config configs/default.yaml \
  --seed "$seed"
