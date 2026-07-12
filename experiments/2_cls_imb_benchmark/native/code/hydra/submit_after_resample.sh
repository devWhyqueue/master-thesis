#!/usr/bin/env bash
# Wait for sample_full jobs to finish, then build WSI caches.
set -euo pipefail

HYDRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HYDRA_DIR"
CONFIG="${1:-config.json}"

echo "Waiting for sample_full jobs to finish..."
while squeue -u "$USER" -h -o "%j" | grep -q '^sample_full$'; do
  sleep 60
done

echo "Constructed splits:"
ls "/home/yannik.qu/master-thesis/prior work/design_dataset/data/constructed_full_scale" | wc -l

bash "${HYDRA_DIR}/submit_wsi_cache_all.sh" "${CONFIG}"
