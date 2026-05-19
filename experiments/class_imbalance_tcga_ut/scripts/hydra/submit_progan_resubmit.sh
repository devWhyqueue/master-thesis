#!/bin/bash
# Regenerate manifests (patches/slide), rebuild SquashFS images, then train ProGAN.
set -euo pipefail
cd "$(dirname "$0")/../.."
export EXPERIMENT_CONTAINER="${EXPERIMENT_CONTAINER:-$PWD/environment.sif}"
export EXPERIMENT_USE_GPU=1
mkdir -p logs
manifest_job=$(sbatch --parsable scripts/hydra/run_patch_manifests.sbatch)
real_sqfs_job=$(sbatch --parsable \
  --dependency="afterok:${manifest_job}" \
  --partition="${PATCH_SQFS_PARTITION:-cpu-5h}" \
  scripts/hydra/build_patch_sqfs.sbatch)
export PROGAN_GAN_DEPENDENCY="afterok:${real_sqfs_job}"
upper="${PROGAN_ARRAY_UPPER:-92}"
parallel="${PROGAN_ARRAY_MAX_PARALLEL:-35}"
gan_job=$(sbatch --parsable \
  --dependency="${PROGAN_GAN_DEPENDENCY}" \
  --array="0-${upper}%${parallel}" \
  --constraint="${PROGAN_GPU_CONSTRAINT:-80gb|40gb|h100}" \
  --partition="${PROGAN_PARTITION:-gpu-5h}" \
  scripts/hydra/run_patch_progan_gan_array.sbatch)
merge_job=$(sbatch --parsable --dependency="afterok:${gan_job}" \
  scripts/hydra/run_merge_progan_manifests.sbatch)
synth_sqfs_job=$(sbatch --parsable \
  --dependency="afterok:${merge_job}" \
  --partition="${PROGAN_SYNTH_SQFS_PARTITION:-cpu-5h}" \
  --array=0-2 \
  scripts/hydra/build_synthetic_sqfs_array.sbatch)
train_job=$(sbatch --parsable \
  --dependency="afterok:${real_sqfs_job}:${synth_sqfs_job}" \
  --constraint="${PROGAN_GPU_CONSTRAINT:-80gb|40gb|h100}" \
  --partition="${PROGAN_TRAIN_PARTITION:-gpu-2d}" \
  scripts/hydra/run_patch_progan_train_array.sbatch)
echo "Patch manifests: ${manifest_job}"
echo "Real SquashFS:   ${real_sqfs_job} (after manifests)"
echo "ProGAN GAN:      ${gan_job} (after real SquashFS, reads via mount)"
echo "ProGAN merge:    ${merge_job}"
echo "Synthetic SquashFS: ${synth_sqfs_job} (after merge)"
echo "ProGAN train:    ${train_job} (after both SquashFS builds)"
