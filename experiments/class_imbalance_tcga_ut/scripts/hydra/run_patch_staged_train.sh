#!/bin/bash
# Stage patch images to node-local storage, then run patch training.
set -euo pipefail
seed="${1:?seed required}"
method="${2:?method required}"
shift 2
include_synthetic=0
if [ "${1:-}" = "--include-synthetic" ]; then
  include_synthetic=1
  shift
fi
export PATCH_STAGE_DIR="${SLURM_TMPDIR:-/tmp}/tcga_ut_patch_seed=${seed}"
stage_args=(--seed "$seed")
if [ "$include_synthetic" = 1 ]; then
  stage_args+=(--include-synthetic)
fi
echo "Staging patch images under ${PATCH_STAGE_DIR}"
STAGED_MANIFEST="$(
  bash scripts/hydra/run_python.sh -m scripts.staging.patch "${stage_args[@]}"
)"
export PATCH_STAGED_MANIFEST="$STAGED_MANIFEST"
echo "Training with staged manifest: ${PATCH_STAGED_MANIFEST}"
bash scripts/hydra/run_python.sh -m scripts.patch.train \
  --method "$method" \
  --seed "$seed" \
  --staged-manifest "$STAGED_MANIFEST" \
  "$@"
