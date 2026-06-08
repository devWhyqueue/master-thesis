#!/bin/bash
# Submit TCGA-UT experiment jobs to the TU Berlin Hydra SLURM cluster.
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

script="scripts/hydra/job.sbatch"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/hydra/submit.sh <command> [job args...]

Commands:
  build-container          Build environment.sif
  prepare                  Build manifests, splits, patch manifests, and exploration output
  build-patch-sqfs         Build the controlled real-patch SquashFS image
  patch-train              Submit non-ProGAN patch benchmark array
  patch-feature-extract    Extract real+synthetic Virchow2 patch feature caches
  patch-feature-train      Train patch-level methods on Virchow2 feature caches
  patch-feature-tune       Submit patch-feature validation-tuning array
  progan                   Submit GAN, synthetic SquashFS, and ProGAN classifier jobs
  progan-resubmit          Regenerate manifests and SquashFS images, then submit ProGAN
  patch-progan-train       Train ProGAN classifier only from existing GAN artifacts
  wsi-cache                Build WSI bag cache array
  wsi-profile              Profile WSI bags
  wsi-train                Submit WSI-bag benchmark array
  wsi-tune                 Submit WSI-bag validation-tuning array
  tuning-aggregate         Aggregate validation-tuning tables and figures
  aggregate                Aggregate tables and figures
  progan-diagnostics       Build ProGAN quality diagnostics for the paper
  all                      Run the full pipeline in one GPU job
  smoke                    Run a short smoke job

Environment overrides:
  EXPERIMENT_CONTAINER, PROGAN_ARRAY_UPPER, PROGAN_ARRAY_MAX_PARALLEL,
  PROGAN_GPU_CONSTRAINT, PROGAN_PARTITION, PROGAN_TRAIN_PARTITION,
  PROGAN_SYNTH_SQFS_PARTITION, PATCH_PARTITION, PATCH_SQFS_PARTITION,
  PATCH_FEATURE_PARTITION, PATCH_FEATURE_CONSTRAINT, WSI_PARTITION
  PATCH_TUNE_PARTITION, WSI_TUNE_PARTITION, TUNING_AGGREGATE_PARTITION
USAGE
}

submit_job() {
  local hydra_job="$1"
  shift
  sbatch --export="ALL,HYDRA_JOB=${hydra_job}" "$@" "$script"
}

require_absent() {
  local path="$1"
  if [ -e "$path" ]; then
    echo "Refusing to submit because target already exists: $path" >&2
    exit 1
  fi
}

require_real_patch_sqfs_absent() {
  require_absent "${PATCH_SQFS_OUTPUT:-/home/space/datasets-sqfs/tcga-ut-controlled-patches.sqfs}"
}

require_synthetic_sqfs_absent() {
  local seed
  for seed in 0 1 2; do
    require_absent "${PATCH_SYNTHETIC_SQFS_OUTPUT:-/home/space/datasets-sqfs/tcga-ut-synthetic-patches-seed-${seed}.sqfs}"
  done
}

submit_with_args() {
  local hydra_job="$1"
  shift
  local separator_seen=0
  local sbatch_args=()
  local job_args=()

  while [ "$#" -gt 0 ]; do
    if [ "$separator_seen" = 0 ] && [ "$1" = "--" ]; then
      separator_seen=1
      shift
      continue
    fi
    if [ "$separator_seen" = 0 ]; then
      sbatch_args+=("$1")
    else
      job_args+=("$1")
    fi
    shift
  done

  sbatch --export="ALL,HYDRA_JOB=${hydra_job}" "${sbatch_args[@]}" "$script" "${job_args[@]}"
}

command="${1:-}"
if [ "$command" = "" ] || [ "$command" = "-h" ] || [ "$command" = "--help" ]; then
  usage
  exit 0
fi
shift

case "$command" in
  build-container)
    submit_job build-container \
      --job-name=tcga-ut-env \
      --partition=cpu-2h \
      --gpus-per-node=0 \
      --ntasks-per-node=2 \
      --output=logs/build-container-%j.out \
      --error=logs/build-container-%j.err \
      "$@"
    ;;

  prepare)
    submit_job prepare \
      --job-name=tcga-ut-prepare \
      --partition=cpu-2h \
      --gpus-per-node=0 \
      --ntasks-per-node=2 \
      --output=logs/prepare-%j.out \
      --error=logs/prepare-%j.err \
      "$@"
    ;;

  build-patch-sqfs)
    require_real_patch_sqfs_absent
    submit_job build-patch-sqfs \
      --job-name=tcga-ut-patch-sqfs \
      --partition="${PATCH_SQFS_PARTITION:-cpu-2h}" \
      --gpus-per-node=0 \
      --ntasks-per-node=4 \
      --output=logs/build-patch-sqfs-%j.out \
      --error=logs/build-patch-sqfs-%j.err \
      "$@"
    ;;

  patch-train)
    submit_with_args patch-train \
      --job-name=tcga-ut-patch \
      --partition="${PATCH_PARTITION:-gpu-2h}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --array=0-17 \
      --output=logs/patch-%A-%a.out \
      --error=logs/patch-%A-%a.err \
      "$@"
    ;;

  patch-progan-train)
    submit_with_args patch-progan-train \
      --job-name=tcga-ut-progan-train \
      --partition="${PROGAN_TRAIN_PARTITION:-gpu-2d}" \
      --constraint="${PROGAN_GPU_CONSTRAINT:-h100|h200|blackwell}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --array=0-2 \
      --output=logs/progan-train-%A-%a.out \
      --error=logs/progan-train-%A-%a.err \
      "$@"
    ;;

  patch-feature-extract)
    submit_with_args patch-feature-extract \
      --job-name=tcga-ut-patch-feat \
      --partition="${PATCH_FEATURE_PARTITION:-gpu-2d}" \
      --constraint="${PATCH_FEATURE_CONSTRAINT:-h100|h200|blackwell}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --array=0-2 \
      --output=logs/patch-feature-extract-%A-%a.out \
      --error=logs/patch-feature-extract-%A-%a.err \
      "$@"
    ;;

  patch-feature-train)
    submit_with_args patch-feature-train \
      --job-name=tcga-ut-patch-feat-train \
      --partition="${PATCH_FEATURE_TRAIN_PARTITION:-gpu-2h}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --array=0-26 \
      --output=logs/patch-feature-train-%A-%a.out \
      --error=logs/patch-feature-train-%A-%a.err \
      "$@"
    ;;

  patch-feature-tune)
    submit_with_args patch-feature-tune \
      --job-name=tcga-ut-patch-tune \
      --partition="${PATCH_TUNE_PARTITION:-gpu-2h}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --array=0-101 \
      --output=logs/patch-tune-%A-%a.out \
      --error=logs/patch-tune-%A-%a.err \
      "$@"
    ;;

  wsi-cache)
    submit_job wsi-cache \
      --job-name=tcga-ut-wsi-cache \
      --partition=cpu-5h \
      --gpus-per-node=0 \
      --ntasks-per-node=4 \
      --array=0-2 \
      --output=logs/wsi-cache-%A-%a.out \
      --error=logs/wsi-cache-%A-%a.err \
      "$@"
    ;;

  wsi-profile)
    submit_job wsi-profile \
      --job-name=tcga-ut-wsi-profile \
      --partition=gpu-test \
      --gpus-per-node=1 \
      --ntasks-per-node=2 \
      --output=logs/wsi-profile-%j.out \
      --error=logs/wsi-profile-%j.err \
      "$@"
    ;;

  wsi-train)
    submit_with_args wsi-train \
      --job-name=tcga-ut-wsi \
      --partition="${WSI_PARTITION:-gpu-2h}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --array=0-20 \
      --output=logs/wsi-%A-%a.out \
      --error=logs/wsi-%A-%a.err \
      "$@"
    ;;

  wsi-tune)
    submit_with_args wsi-tune \
      --job-name=tcga-ut-wsi-tune \
      --partition="${WSI_TUNE_PARTITION:-gpu-2h}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --array=0-95 \
      --output=logs/wsi-tune-%A-%a.out \
      --error=logs/wsi-tune-%A-%a.err \
      "$@"
    ;;

  tuning-aggregate)
    submit_with_args tuning-aggregate \
      --job-name=tcga-ut-tune-agg \
      --partition="${TUNING_AGGREGATE_PARTITION:-gpu-9m}" \
      --gpus-per-node=1 \
      --ntasks-per-node=2 \
      --output=logs/tuning-aggregate-%j.out \
      --error=logs/tuning-aggregate-%j.err \
      "$@"
    ;;

  aggregate)
    submit_job aggregate \
      --job-name=tcga-ut-aggregate \
      --partition=cpu-2h \
      --gpus-per-node=0 \
      --ntasks-per-node=2 \
      --output=logs/aggregate-%j.out \
      --error=logs/aggregate-%j.err \
      "$@"
    ;;

  progan-diagnostics)
    submit_job progan-diagnostics \
      --job-name=tcga-ut-progan-diag \
      --partition=cpu-2h \
      --gpus-per-node=0 \
      --ntasks-per-node=4 \
      --output=logs/progan-diagnostics-%j.out \
      --error=logs/progan-diagnostics-%j.err \
      "$@"
    ;;

  all)
    submit_with_args all \
      --job-name=tcga-ut-imbalance \
      --partition=gpu-2h \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --output=logs/all-%j.out \
      --error=logs/all-%j.err \
      "$@"
    ;;

  smoke)
    submit_with_args smoke \
      --job-name=tcga-ut-smoke \
      --partition=gpu-test \
      --gpus-per-node=1 \
      --ntasks-per-node=2 \
      --output=logs/smoke-%j.out \
      --error=logs/smoke-%j.err \
      "$@"
    ;;

  progan)
    require_synthetic_sqfs_absent
    upper="${PROGAN_ARRAY_UPPER:-92}"
    parallel="${PROGAN_ARRAY_MAX_PARALLEL:-35}"
    gan_deps=()
    if [ -n "${PROGAN_GAN_DEPENDENCY:-}" ]; then
      gan_deps=(--dependency="${PROGAN_GAN_DEPENDENCY}")
    fi
    gan_job=$(sbatch --parsable \
      --export=ALL,HYDRA_JOB=patch-progan-gan \
      "${gan_deps[@]}" \
      --array="0-${upper}%${parallel}" \
      --constraint="${PROGAN_GPU_CONSTRAINT:-h100|h200|blackwell}" \
      --partition="${PROGAN_PARTITION:-gpu-5h}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --job-name=tcga-ut-progan-gan \
      --output=logs/progan-gan-%A-%a.out \
      --error=logs/progan-gan-%A-%a.err \
      "$script")
    sqfs_job=$(sbatch --parsable \
      --export=ALL,HYDRA_JOB=build-synthetic-sqfs \
      --dependency="afterok:${gan_job}" \
      --partition="${PROGAN_SYNTH_SQFS_PARTITION:-cpu-5h}" \
      --gpus-per-node=0 \
      --ntasks-per-node=4 \
      --array=0-2 \
      --job-name=tcga-ut-synth-sqfs \
      --output=logs/build-synthetic-sqfs-%A-%a.out \
      --error=logs/build-synthetic-sqfs-%A-%a.err \
      "$script")
    train_job=$(sbatch --parsable \
      --export=ALL,HYDRA_JOB=patch-progan-train \
      --dependency="afterok:${sqfs_job}" \
      --constraint="${PROGAN_GPU_CONSTRAINT:-h100|h200|blackwell}" \
      --partition="${PROGAN_TRAIN_PARTITION:-gpu-2d}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --array=0-2 \
      --job-name=tcga-ut-progan-train \
      --output=logs/progan-train-%A-%a.out \
      --error=logs/progan-train-%A-%a.err \
      "$script")
    echo "ProGAN GAN array: ${gan_job} (tasks 0-${upper}, max ${parallel} parallel)"
    echo "Synthetic SquashFS array: ${sqfs_job} (after GAN, seeds 0-2)"
    echo "ProGAN train array: ${train_job} (after synthetic SquashFS build)"
    ;;

  progan-resubmit)
    require_real_patch_sqfs_absent
    require_synthetic_sqfs_absent
    manifest_job=$(sbatch --parsable \
      --export=ALL,HYDRA_JOB=patch-manifests \
      --partition=cpu-2h \
      --gpus-per-node=0 \
      --ntasks-per-node=2 \
      --job-name=tcga-ut-patch-manifests \
      --output=logs/patch-manifests-%j.out \
      --error=logs/patch-manifests-%j.err \
      "$script")
    real_sqfs_job=$(sbatch --parsable \
      --export=ALL,HYDRA_JOB=build-patch-sqfs \
      --dependency="afterok:${manifest_job}" \
      --partition="${PATCH_SQFS_PARTITION:-cpu-5h}" \
      --gpus-per-node=0 \
      --ntasks-per-node=4 \
      --job-name=tcga-ut-patch-sqfs \
      --output=logs/build-patch-sqfs-%j.out \
      --error=logs/build-patch-sqfs-%j.err \
      "$script")
    upper="${PROGAN_ARRAY_UPPER:-92}"
    parallel="${PROGAN_ARRAY_MAX_PARALLEL:-35}"
    gan_job=$(sbatch --parsable \
      --export=ALL,HYDRA_JOB=patch-progan-gan \
      --dependency="afterok:${real_sqfs_job}" \
      --array="0-${upper}%${parallel}" \
      --constraint="${PROGAN_GPU_CONSTRAINT:-h100|h200|blackwell}" \
      --partition="${PROGAN_PARTITION:-gpu-5h}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --job-name=tcga-ut-progan-gan \
      --output=logs/progan-gan-%A-%a.out \
      --error=logs/progan-gan-%A-%a.err \
      "$script")
    merge_job=$(sbatch --parsable \
      --export=ALL,HYDRA_JOB=merge-progan-manifests \
      --dependency="afterok:${gan_job}" \
      --partition=cpu-2h \
      --gpus-per-node=0 \
      --ntasks-per-node=2 \
      --job-name=tcga-ut-progan-merge \
      --output=logs/progan-merge-%j.out \
      --error=logs/progan-merge-%j.err \
      "$script")
    synth_sqfs_job=$(sbatch --parsable \
      --export=ALL,HYDRA_JOB=build-synthetic-sqfs \
      --dependency="afterok:${merge_job}" \
      --partition="${PROGAN_SYNTH_SQFS_PARTITION:-cpu-5h}" \
      --gpus-per-node=0 \
      --ntasks-per-node=4 \
      --array=0-2 \
      --job-name=tcga-ut-synth-sqfs \
      --output=logs/build-synthetic-sqfs-%A-%a.out \
      --error=logs/build-synthetic-sqfs-%A-%a.err \
      "$script")
    train_job=$(sbatch --parsable \
      --export=ALL,HYDRA_JOB=patch-progan-train \
      --dependency="afterok:${real_sqfs_job}:${synth_sqfs_job}" \
      --constraint="${PROGAN_GPU_CONSTRAINT:-h100|h200|blackwell}" \
      --partition="${PROGAN_TRAIN_PARTITION:-gpu-2d}" \
      --gpus-per-node=1 \
      --ntasks-per-node=8 \
      --array=0-2 \
      --job-name=tcga-ut-progan-train \
      --output=logs/progan-train-%A-%a.out \
      --error=logs/progan-train-%A-%a.err \
      "$script")
    echo "Patch manifests: ${manifest_job}"
    echo "Real SquashFS:   ${real_sqfs_job} (after manifests)"
    echo "ProGAN GAN:      ${gan_job} (after real SquashFS, reads via mount)"
    echo "ProGAN merge:    ${merge_job}"
    echo "Synthetic SquashFS: ${synth_sqfs_job} (after merge)"
    echo "ProGAN train:    ${train_job} (after both SquashFS builds)"
    ;;

  *)
    echo "Unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
