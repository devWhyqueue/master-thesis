#!/bin/bash
#SBATCH --job-name=imb-check
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/yannik.qu/master-thesis/experiments/01_benchmark/02_benchmark_patch/outputs/logs/%x-%j.out
#SBATCH --error=/home/yannik.qu/master-thesis/experiments/01_benchmark/02_benchmark_patch/outputs/logs/%x-%j.err
set -euo pipefail
ROOT=/home/yannik.qu/master-thesis
P=$ROOT/experiments/03_class_imbalance/01_cause/cross/48_lora_decomposition
mkdir -p "$P/report"
export APPTAINERENV_PYTHONPATH="$P/code"
apptainer exec -B "$ROOT:$ROOT:ro" -B "$P/report:$P/report:rw" -B "$P/outputs:$P/outputs:rw" -B "/home/space:/home/space:ro" \
  "$ROOT/experiments/environment.sif" \
  bash -lc "python $P/code/__main__.py --config $P/configs/bracs.yaml --config $P/configs/tcga_ut.yaml check --phase $PHASE"
