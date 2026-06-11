#!/usr/bin/env bash
set -euo pipefail

HYDRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${HYDRA_DIR}/.." && pwd)"
cd "$HYDRA_DIR"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
CONFIG="${1:-config.json}"
PYTHON="python3 run.py --config=${CONFIG}"

ORDERS=(easy_to_difficult difficult_to_easy)
PARAMS=(0.0 1.0 1.3)
SEEDS=(0 1 2)
CLASS_ORDER_DIR="/home/yannik.qu/master-thesis/prior work/design_dataset/report/outputs/class_orders"

for order in "${ORDERS[@]}"; do
  for param in "${PARAMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "resample order=${order} param=${param} seed=${seed}"
      ${PYTHON} sample-full-scale \
        --parameter="${param}" \
        --seed="${seed}" \
        --class-order-name="${order}" \
        --class-order-file="${CLASS_ORDER_DIR}/${order}.json"
    done
  done
done

echo "Resampling submitted. After jobs finish, run: bash submit_wsi_cache_all.sh ${CONFIG}"
