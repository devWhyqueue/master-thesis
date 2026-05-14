#!/bin/bash
#SBATCH --job-name=tcga-ut-enc
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-2%1
#SBATCH --output=logs/synthetic-encode-%A-%a.out
#SBATCH --error=logs/synthetic-encode-%A-%a.err

set -euo pipefail

if [ -f configs/default.yaml ]; then
  cd .
elif [ -f experiments/class_imbalance_tcga_ut/configs/default.yaml ]; then
  cd experiments/class_imbalance_tcga_ut
else
  echo "Submit from the experiment directory or repository root." >&2
  exit 1
fi

if [ "${HF_TOKEN:-}" = "" ] && [ "${HUGGINGFACE_HUB_TOKEN:-}" = "" ]; then
  echo "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN for gated paige-ai/Virchow2 access." >&2
  exit 1
fi

seeds=(0 1 2)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

export EXPERIMENT_USE_GPU=1
bash scripts/hydra/run_python.sh -m scripts.synthetic.encode_virchow2_features \
  --config configs/default.yaml \
  --seed "$seed"
