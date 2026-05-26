#!/bin/bash
#SBATCH --job-name=tcga-ut
#SBATCH --partition=cpu-test
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

job="${HYDRA_JOB:?Set HYDRA_JOB to the job name to run.}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/configs/default.yaml" ]; then
  ROOT="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/experiments/class_imbalance/configs/default.yaml" ]; then
  ROOT="${SLURM_SUBMIT_DIR}/experiments/class_imbalance"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

cd "$ROOT"
mkdir -p logs
export EXPERIMENT_CONTAINER="${EXPERIMENT_CONTAINER:-$ROOT/environment.sif}"

seed_from_array() {
  local seeds=(0 1 2)
  printf "%s" "${seeds[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}]}"
}

case "$job" in
  build-container)
    apptainer build --force environment.sif configs/environment.def
    ;;

  build-patch-sqfs)
    export PATCH_SQFS_OUTPUT="${PATCH_SQFS_OUTPUT:-/home/space/datasets-sqfs/tcga-ut-controlled-patches.sqfs}"
    export PATCH_SQFS_FILE_LIST="${PATCH_SQFS_FILE_LIST:-${SLURM_TMPDIR:-/tmp}/patch_sqfs_files.txt}"
    if [ -e "$PATCH_SQFS_OUTPUT" ]; then
      echo "Refusing to overwrite existing SquashFS image: $PATCH_SQFS_OUTPUT" >&2
      exit 1
    fi
    if ! command -v mksquashfs >/dev/null 2>&1; then
      echo "mksquashfs not found on PATH" >&2
      exit 1
    fi
    if ! command -v tar >/dev/null 2>&1; then
      echo "tar not found on PATH" >&2
      exit 1
    fi
    raw_root="$(bash scripts/hydra/run_python.sh -m scripts.prep.export_patch_sqfs_input --file-list "$PATCH_SQFS_FILE_LIST")"
    tar -C "$raw_root" -cf - -T "$PATCH_SQFS_FILE_LIST" | mksquashfs - "$PATCH_SQFS_OUTPUT" \
      -tar \
      -noappend \
      -comp "${PATCH_SQFS_COMP:-lz4}" \
      -no-xattrs \
      -no-duplicates \
      -processors "${PATCH_SQFS_PROCESSORS:-${SLURM_NTASKS:-4}}" \
      -no-progress
    echo "Wrote SquashFS image: $PATCH_SQFS_OUTPUT"
    ;;

  build-synthetic-sqfs)
    seed="$(seed_from_array)"
    export PATCH_SYNTHETIC_SQFS_OUTPUT="${PATCH_SYNTHETIC_SQFS_OUTPUT:-/home/space/datasets-sqfs/tcga-ut-synthetic-patches-seed-${seed}.sqfs}"
    export PATCH_SYNTHETIC_SQFS_FILE_LIST="${PATCH_SYNTHETIC_SQFS_FILE_LIST:-${SLURM_TMPDIR:-/tmp}/tcga_ut_synthetic_seed=${seed}.txt}"
    if [ -e "$PATCH_SYNTHETIC_SQFS_OUTPUT" ]; then
      echo "Refusing to overwrite existing SquashFS image: $PATCH_SYNTHETIC_SQFS_OUTPUT" >&2
      exit 1
    fi
    if ! command -v mksquashfs >/dev/null 2>&1; then
      echo "mksquashfs not found on PATH" >&2
      exit 1
    fi
    if ! command -v tar >/dev/null 2>&1; then
      echo "tar not found on PATH" >&2
      exit 1
    fi
    image_root="$(bash scripts/hydra/run_python.sh -m scripts.prep.export_synthetic_sqfs_input --seed "$seed" --file-list "$PATCH_SYNTHETIC_SQFS_FILE_LIST")"
    tar -C "$image_root" -cf - -T "$PATCH_SYNTHETIC_SQFS_FILE_LIST" | mksquashfs - "$PATCH_SYNTHETIC_SQFS_OUTPUT" \
      -tar \
      -noappend \
      -comp "${PATCH_SQFS_COMP:-lz4}" \
      -no-xattrs \
      -no-duplicates \
      -processors "${PATCH_SQFS_PROCESSORS:-${SLURM_NTASKS:-4}}" \
      -no-progress
    echo "Wrote synthetic SquashFS for seed=${seed}: ${PATCH_SYNTHETIC_SQFS_OUTPUT}"
    ;;

  prepare)
    bash scripts/hydra/run_python.sh -m scripts.prep.check_env
    bash scripts/hydra/run_python.sh -m scripts.prep.manifest
    for seed in 0 1 2; do
      bash scripts/hydra/run_python.sh -m scripts.prep.splits --seed "$seed"
      bash scripts/hydra/run_python.sh -m scripts.prep.patch_manifest --seed "$seed"
    done
    bash scripts/hydra/run_python.sh -m scripts.prep.explore --seed 0
    ;;

  patch-manifests)
    for seed in 0 1 2; do
      bash scripts/hydra/run_python.sh -m scripts.prep.patch_manifest --seed "$seed"
    done
    ;;

  patch-train)
    export EXPERIMENT_USE_GPU=1
    methods=(patch_ce patch_weighted_ce patch_focal patch_balanced_sampler_ce patch_ce_soft_f1_balanced patch_ce_soft_mcc_balanced)
    seeds=(0 1 2)
    method="${methods[$((SLURM_ARRAY_TASK_ID / 3))]}"
    seed="${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}"
    bash scripts/hydra/run_patch_staged_train.sh "$seed" "$method" "$@"
    ;;

  patch-progan-gan)
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
      --array-task-id "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}" \
      "$@"
    ;;

  merge-progan-manifests)
    for seed in 0 1 2; do
      bash scripts/hydra/run_python.sh -m scripts.patch.synthetic --seed "$seed" --merge-only
    done
    ;;

  patch-progan-train)
    export EXPERIMENT_USE_GPU=1
    seed="$(seed_from_array)"
    bash scripts/hydra/run_patch_staged_train.sh "$seed" patch_progan_aug \
      --include-synthetic \
      --skip-synthetic-generation \
      --resume \
      "$@"
    ;;

  patch-feature-extract)
    export EXPERIMENT_USE_GPU=1
    seed="$(seed_from_array)"
    bash scripts/hydra/run_patch_feature_extract.sh "$seed" "$@"
    ;;

  patch-feature-train)
    export EXPERIMENT_USE_GPU=1
    methods=(patch_feature_ce patch_feature_weighted_ce patch_feature_focal patch_feature_balanced_sampler_ce patch_feature_ce_soft_f1_balanced patch_feature_ce_soft_mcc_balanced patch_feature_progan_aug patch_feature_cfal patch_feature_divide_conquer)
    seeds=(0 1 2)
    method="${methods[$((SLURM_ARRAY_TASK_ID / 3))]}"
    seed="${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}"
    bash scripts/hydra/run_python.sh -m scripts.train_patch_features --method "$method" --seed "$seed" "$@"
    ;;

  patch-feature-tune)
    export EXPERIMENT_USE_GPU=1
    bash scripts/hydra/run_python.sh -m scripts.tuning.run \
      --benchmark patch_feature \
      --array-task-id "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}" \
      "$@"
    ;;

  wsi-cache)
    seed="$(seed_from_array)"
    bash scripts/hydra/run_python.sh -m scripts.prep.wsi_bag_cache --seed "$seed"
    ;;

  wsi-profile)
    export EXPERIMENT_USE_GPU=1
    bash scripts/hydra/run_python.sh -m scripts.profile_wsi_bags --seed 0
    ;;

  wsi-train)
    export EXPERIMENT_USE_GPU=1
    methods=(mil_ce mil_weighted_ce mil_focal mil_balanced_sampler_ce rankmix_mil sc_mil mde_mil)
    seeds=(0 1 2)
    method="${methods[$((SLURM_ARRAY_TASK_ID / 3))]}"
    seed="${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}"
    bash scripts/hydra/run_python.sh -m scripts.training.train --method "$method" --seed "$seed" "$@"
    ;;

  wsi-tune)
    export EXPERIMENT_USE_GPU=1
    bash scripts/hydra/run_python.sh -m scripts.tuning.run \
      --benchmark wsi_bag \
      --array-task-id "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}" \
      "$@"
    ;;

  tuning-aggregate)
    export EXPERIMENT_USE_GPU=1
    bash scripts/hydra/run_python.sh -m scripts.tuning.aggregate "$@"
    ;;

  aggregate)
    bash scripts/hydra/run_python.sh -m scripts.report.recompute_tier_metrics
    bash scripts/hydra/run_python.sh -m scripts.report.aggregate --benchmark patch
    bash scripts/hydra/run_python.sh -m scripts.report.aggregate --benchmark wsi_bag
    bash scripts/hydra/run_python.sh -m scripts.report.calibration_table
    bash scripts/hydra/run_python.sh -m scripts.report.calibration_posthoc_table
    bash scripts/hydra/run_python.sh -m scripts.report.posthoc_calibration --seed 0 --calibrators temperature
    bash scripts/hydra/run_python.sh -m scripts.report.paired_delta_table
    bash scripts/hydra/run_python.sh -m scripts.report.progan_diagnostics --seed 0
    bash scripts/hydra/run_python.sh -m scripts.report.figures --benchmark patch
    bash scripts/hydra/run_python.sh -m scripts.report.figures --benchmark wsi_bag
    ;;

  progan-diagnostics)
    # shellcheck source=scripts/hydra/mount_real_patch_sqfs.sh
    source scripts/hydra/mount_real_patch_sqfs.sh direct
    trap cleanup_real_patch_sqfs EXIT
    bash scripts/hydra/run_python.sh -m scripts.report.progan_diagnostics --seed 0 "$@"
    ;;

  all)
    export EXPERIMENT_USE_GPU=1
    bash scripts/hydra/run_python.sh -m scripts.run_pipeline --benchmark all "$@"
    ;;

  smoke)
    export EXPERIMENT_USE_GPU=1
    bash scripts/hydra/run_python.sh -m scripts.run_pipeline --benchmark all --smoke "$@"
    ;;

  *)
    echo "Unknown HYDRA_JOB: $job" >&2
    exit 2
    ;;
esac
