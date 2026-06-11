#!/usr/bin/env bash
# Continue aligned workflow after native_prevalence sampling/cache.
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
REPORT_ROOT="/home/yannik.qu/master-thesis/prior work/design_dataset/report/outputs"
CLASS_ORDER_DIR="${REPORT_ROOT}/class_orders"

echo "=== Sample missing class orders ==="
for order in "${ORDERS[@]}"; do
  for param in "${PARAMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "sample-full-scale order=${order} param=${param} seed=${seed}"
      ${PYTHON} sample-full-scale \
        --parameter="${param}" \
        --seed="${seed}" \
        --class-order-name="${order}" \
        --class-order-file="${CLASS_ORDER_DIR}/${order}.json"
    done
  done
done

echo "=== WSI bag cache (all 27 splits) ==="
ALL_ORDERS=(native_prevalence easy_to_difficult difficult_to_easy)
for order in "${ALL_ORDERS[@]}"; do
  for param in "${PARAMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "wsi-cache order=${order} param=${param} seed=${seed}"
      ${PYTHON} wsi-cache \
        --parameter="${param}" \
        --seed="${seed}" \
        --class-order-name="${order}"
    done
  done
done

echo "=== Validation tuning ==="
${PYTHON} tune
${PYTHON} tune-wsi

echo "When tuning finishes: bash submit_post_tuning.sh ${CONFIG}"
