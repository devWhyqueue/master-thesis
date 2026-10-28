#!/bin/bash
#SBATCH --job-name=exp46
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/yannik.qu/master-thesis/experiments/03_class_imbalance/02_mitigation/46_ts_noninferiority/logs/exp46-%j.out
set -euo pipefail
ROOT=/home/yannik.qu/master-thesis
EXP=$ROOT/experiments/03_class_imbalance/02_mitigation/46_ts_noninferiority
cd "$ROOT"
run() { apptainer run -B /home/space:/home/space:rw "$ROOT/experiments/environment.sif" python3 "$EXP/code" --config "$EXP/configs/ts_nll.yaml" "$@"; }
run analyze
