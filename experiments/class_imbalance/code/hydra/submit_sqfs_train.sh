#!/bin/bash
# Rebuild r2 synthetic SQFS for seeds 0 and 2 (seed 1 already valid).
# Then chain progan-train on all three seeds.
set -euo pipefail
cd /home/yannik.qu/master-thesis/experiments/class_imbalance

SQFS_TEMPLATE="/home/space/datasets-sqfs/tcga-ut-synthetic-patches-r2-seed-{seed}.sqfs"

sqfs_job=$(sbatch --parsable \
  --export="ALL,HYDRA_JOB=build-synthetic-sqfs,PATCH_SYNTHETIC_SQFS_TEMPLATE=${SQFS_TEMPLATE},PATCH_SQFS_PROCESSORS=2" \
  --partition=cpu-2d \
  --gpus-per-node=0 \
  --ntasks-per-node=2 \
  --mem=256G \
  --array=0,2 \
  --job-name=tcga-ut-synth-sqfs \
  --output=logs/build-synthetic-sqfs-%A-%a.out \
  --error=logs/build-synthetic-sqfs-%A-%a.err \
  scripts/hydra/job.sbatch)
echo "SQFS build (seeds 0,2): ${sqfs_job}"

# Seed 1 SQFS already valid; still include seed 1 in train array (it will use the existing file)
train_job=$(sbatch --parsable \
  --export="ALL,HYDRA_JOB=patch-progan-train,PATCH_SYNTHETIC_SQFS_TEMPLATE=${SQFS_TEMPLATE}" \
  --dependency="afterok:${sqfs_job}" \
  --constraint="40gb|80gb|h100|h200|blackwell" \
  --partition=gpu-5h \
  --gpus-per-node=1 \
  --ntasks-per-node=8 \
  --array=0-2 \
  --job-name=tcga-ut-progan-train \
  --output=logs/progan-train-%A-%a.out \
  --error=logs/progan-train-%A-%a.err \
  scripts/hydra/job.sbatch)
echo "Train (seeds 0-2):      ${train_job} (after ${sqfs_job})"
