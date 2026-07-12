#!/usr/bin/env bash
# Run after patch/WSI tuning jobs have finished.
set -euo pipefail

HYDRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${HYDRA_DIR}/.." && pwd)"
cd "$HYDRA_DIR"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
CONFIG="${1:-config.json}"
PYTHON="python3 run.py --config=${CONFIG}"

echo "=== Tuning aggregate ==="
${PYTHON} tune-aggregate
# Wait for tune_aggregate to finish before submitting report (report reads tuning_selection.json)
while squeue -u "$(whoami)" -n tune_aggregate -h | grep -q .; do
  sleep 30
done

echo "=== Report tables ==="
${PYTHON} report

echo "Done."
