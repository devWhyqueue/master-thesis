#!/bin/bash
# Backup SQFS builds on cpu-2d (idempotent — skips rebuild if already valid).
# Chains training on the e25 backup build.
set -euo pipefail
cd /home/yannik.qu/master-thesis/experiments/class_imbalance

SQFS_BASE=/home/space/datasets-sqfs

build_epoch() {
  local epoch="$1"
  sbatch --parsable \
    --export="ALL,HYDRA_JOB=build-synthetic-sqfs,PATCH_SYNTHETIC_SQFS_TEMPLATE=${SQFS_BASE}/tcga-ut-synthetic-patches-e${epoch}-seed-{seed}.sqfs,PATCH_SYNTHETIC_SQFS_EPOCH_REF=${epoch},PATCH_SQFS_PROCESSORS=4" \
    --partition=cpu-2d \
    --gpus-per-node=0 \
    --ntasks-per-node=4 \
    --mem=64G \
    --array=0-2 \
    --job-name=tcga-ut-synth-sqfs-e${epoch}-bak \
    --output=logs/build-synthetic-sqfs-e${epoch}-bak-%A-%a.out \
    --error=logs/build-synthetic-sqfs-e${epoch}-bak-%A-%a.err \
    scripts/hydra/job.sbatch
}

e10=$(build_epoch 10); echo "Backup build e10: ${e10}"
e25=$(build_epoch 25); echo "Backup build e25: ${e25}"
e50=$(build_epoch 50); echo "Backup build e50: ${e50}"

train_job=$(sbatch --parsable \
  --export="ALL,HYDRA_JOB=patch-progan-train,PATCH_SYNTHETIC_SQFS_TEMPLATE=${SQFS_BASE}/tcga-ut-synthetic-patches-e25-seed-{seed}.sqfs" \
  --dependency="afterok:${e25}" \
  --constraint="40gb|80gb|h100|h200|blackwell" \
  --partition=gpu-5h \
  --gpus-per-node=1 \
  --ntasks-per-node=8 \
  --array=0-2 \
  --job-name=tcga-ut-progan-train \
  --output=logs/progan-train-%A-%a.out \
  --error=logs/progan-train-%A-%a.err \
  scripts/hydra/job.sbatch)
echo "Train (seeds 0-2):  ${train_job} (after e25 backup ${e25})"
