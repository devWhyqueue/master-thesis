#!/usr/bin/env bash
set -euo pipefail

HYDRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${HYDRA_DIR}/.." && pwd)"
cd "$HYDRA_DIR"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
CONFIG="${1:-config.json}"
PYTHON="python3 run.py --config=${CONFIG}"

ORDERS=(native_prevalence easy_to_difficult difficult_to_easy)
PARAMS=(0.0 1.0 1.3)
SEEDS=(0 1 2)

for order in "${ORDERS[@]}"; do
  for param in "${PARAMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      ${PYTHON} wsi-cache --parameter="${param}" --seed="${seed}" --class-order-name="${order}"
    done
  done
done

echo "WSI cache jobs submitted."
