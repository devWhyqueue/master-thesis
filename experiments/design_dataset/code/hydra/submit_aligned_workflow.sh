#!/usr/bin/env bash
# Submit the aligned prior-work pipeline (class_imbalance methodology).
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
REPORT_ROOT="/home/yannik.qu/master-thesis/prior work/design_dataset/report/outputs"
CLASS_ORDER_DIR="${REPORT_ROOT}/class_orders"

echo "=== Cancel obsolete prior-work jobs ==="
scancel --name=wsi_train -u "$USER" 2>/dev/null || true
scancel --name=constructed-report -u "$USER" 2>/dev/null || true
scancel --name=construc -u "$USER" 2>/dev/null || true

echo "=== Phase 1: constructed sampling (27 regimes x 3 seeds) ==="
for order in "${ORDERS[@]}"; do
  extra=()
  if [[ "${order}" != "native_prevalence" ]]; then
    extra=(--class-order-file="${CLASS_ORDER_DIR}/${order}.json")
  fi
  for param in "${PARAMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "sample-full-scale order=${order} param=${param} seed=${seed}"
      ${PYTHON} sample-full-scale \
        --parameter="${param}" \
        --seed="${seed}" \
        --class-order-name="${order}" \
        ${extra[@]+"${extra[@]}"}
    done
  done
done

echo "=== Phase 2: verify cls_patchmean features ==="
${PYTHON} verify-features

echo "=== Phase 3: WSI bag cache (27 constructed splits) ==="
for order in "${ORDERS[@]}"; do
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

echo "=== Phase 4: validation tuning (patch + WSI) ==="
${PYTHON} tune
${PYTHON} tune-wsi

echo "=== Phase 4 submitted patch + WSI tuning jobs ==="
echo "When tuning queue is empty, run: bash submit_post_tuning.sh ${CONFIG}"
echo "Monitor with: squeue -u ${USER}"
