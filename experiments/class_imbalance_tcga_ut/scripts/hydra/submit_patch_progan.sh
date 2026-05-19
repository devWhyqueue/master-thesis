#!/bin/bash
# Submit parallel ProGAN GAN jobs, then classifier training after they finish.
set -euo pipefail
cd "$(dirname "$0")/../.."
export EXPERIMENT_USE_GPU=1
upper="${PROGAN_ARRAY_UPPER:-92}"
parallel="${PROGAN_ARRAY_MAX_PARALLEL:-35}"
gan_deps=()
if [ -n "${PROGAN_GAN_DEPENDENCY:-}" ]; then
  gan_deps=(--dependency="${PROGAN_GAN_DEPENDENCY}")
fi
gan_job=$(sbatch --parsable \
  "${gan_deps[@]}" \
  --array="0-${upper}%${parallel}" \
  --constraint="${PROGAN_GPU_CONSTRAINT:-80gb|40gb|h100}" \
  --partition="${PROGAN_PARTITION:-gpu-5h}" \
  scripts/hydra/run_patch_progan_gan_array.sbatch)
sqfs_job=$(sbatch --parsable \
  --dependency="afterok:${gan_job}" \
  --partition="${PROGAN_SYNTH_SQFS_PARTITION:-cpu-5h}" \
  --array=0-2 \
  scripts/hydra/build_synthetic_sqfs_array.sbatch)
train_job=$(sbatch --parsable \
  --dependency="afterok:${sqfs_job}" \
  --constraint="${PROGAN_GPU_CONSTRAINT:-80gb|40gb|h100}" \
  --partition="${PROGAN_TRAIN_PARTITION:-gpu-2d}" \
  scripts/hydra/run_patch_progan_train_array.sbatch)
echo "ProGAN GAN array: ${gan_job} (tasks 0-${upper}, max ${parallel} parallel)"
echo "Synthetic SquashFS array: ${sqfs_job} (after GAN, seeds 0-2)"
echo "ProGAN train array: ${train_job} (after synthetic SquashFS build)"
