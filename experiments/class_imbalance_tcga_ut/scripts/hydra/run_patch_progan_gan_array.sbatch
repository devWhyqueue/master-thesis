#!/bin/bash
#SBATCH --job-name=tcga-ut-progan-gan
#SBATCH --partition=gpu-5h
#SBATCH --constraint=80gb|40gb|h100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/progan-gan-%A-%a.out
#SBATCH --error=logs/progan-gan-%A-%a.err
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
export EXPERIMENT_USE_GPU=1
# shellcheck source=scripts/hydra/mount_real_patch_sqfs.sh
source scripts/hydra/mount_real_patch_sqfs.sh direct
trap cleanup_real_patch_sqfs EXIT
if [ -n "${PATCH_SQFS_MOUNT:-}" ]; then
  echo "GAN training reads real patches from ${PATCH_SQFS_MOUNT}"
else
  echo "SquashFS unavailable; GAN reads real patches from BeeGFS"
fi
bash scripts/hydra/run_python.sh -m scripts.patch.synthetic \
  --array-task-id "${SLURM_ARRAY_TASK_ID}" \
  "$@"
