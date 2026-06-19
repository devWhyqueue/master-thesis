"""Submit the full ProGAN rerun chain with dependency chaining.

B4 submission: progan GAN → feature-extract → feature-train (progan only)
             → feature-tune (progan only) → tuning-aggregate → aggregate
             + progan-diagnostics

Run with optional --existing-sqfs=<job_id> to skip the GAN/merge/sqfs
submission when those jobs are already queued from a prior run.
"""
import os
import subprocess
import sys


def sbatch(args: list[str], *, dependency: str | None = None) -> str:
    dep_flags = [f"--dependency={dependency}"] if dependency else []
    # dependency must come before the script path (last element of args)
    script = args[-1]
    sbatch_opts = args[:-1]
    cmd = ["sbatch", "--parsable"] + dep_flags + sbatch_opts + [script]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def run(cmd: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(cmd, capture_output=capture, text=True, check=True)
    return result.stdout if capture else ""


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> None:
    script = "scripts/hydra/job.sbatch"

    existing_sqfs = None
    for arg in sys.argv[1:]:
        if arg.startswith("--existing-sqfs="):
            existing_sqfs = arg.split("=", 1)[1]

    if existing_sqfs:
        sqfs_id = existing_sqfs
        print(f"Reusing existing sqfs job id: {sqfs_id}")
    else:
        # ── Step 1: ProGAN GAN + merge + SquashFS chain ──────────────────────
        print("Submitting ProGAN chain (GAN → merge → SquashFS) …")
        progan_out = run(
            ["bash", "scripts/hydra/submit.sh", "progan"],
            capture=True,
        )
        print(progan_out)

        sqfs_id = next(
            line.split(":")[1].strip().split()[0]
            for line in progan_out.splitlines()
            if "Synthetic SquashFS" in line
        )
        train_job_id = next(
            line.split(":")[1].strip().split()[0]
            for line in progan_out.splitlines()
            if "ProGAN train" in line
        )
        print(f"  sqfs_id={sqfs_id}  progan-train={train_job_id} (will cancel)")

        # Cancel the image-classifier ProGAN train (not in the report)
        subprocess.run(["scancel", train_job_id], check=True)
        print(f"  Cancelled patch-progan-train job {train_job_id}")

    # ── Step 2: Feature extraction (real + new synthetic, all seeds) ─────────
    print("Submitting patch-feature-extract (array 0-2) …")
    feat_id = sbatch(
        [
            "--export=ALL,HYDRA_JOB=patch-feature-extract",
            "--job-name=tcga-ut-patch-feat",
            f"--partition={_env('PATCH_FEATURE_PARTITION', 'gpu-2d')}",
            f"--constraint={_env('PATCH_FEATURE_CONSTRAINT', 'h100|h200|blackwell')}",
            "--gpus-per-node=1",
            "--ntasks-per-node=8",
            "--array=0-2",
            "--output=logs/patch-feature-extract-%A-%a.out",
            "--error=logs/patch-feature-extract-%A-%a.err",
            script,
        ],
        dependency=f"afterok:{sqfs_id}",
    )
    print(f"  feat_extract_id={feat_id}")

    # ── Step 3a: Train patch_feature_progan_aug (tasks 18-20) ────────────────
    print("Submitting patch-feature-train --array=18-20 …")
    feat_train_id = sbatch(
        [
            "--export=ALL,HYDRA_JOB=patch-feature-train",
            "--job-name=tcga-ut-pf-train-progan",
            f"--partition={_env('PATCH_FEATURE_TRAIN_PARTITION', 'gpu-2h')}",
            "--gpus-per-node=1",
            "--ntasks-per-node=8",
            "--array=18-20",
            "--output=logs/patch-feature-train-%A-%a.out",
            "--error=logs/patch-feature-train-%A-%a.err",
            script,
        ],
        dependency=f"afterok:{feat_id}",
    )
    print(f"  feat_train_id={feat_train_id}")

    # ── Step 3b: Tune ProGAN final-depth-epoch sweep (tasks 111-119) ─────────
    print("Submitting patch-feature-tune --array=111-119 …")
    feat_tune_id = sbatch(
        [
            "--export=ALL,HYDRA_JOB=patch-feature-tune",
            "--job-name=tcga-ut-pf-tune-progan",
            f"--partition={_env('PATCH_TUNE_PARTITION', 'gpu-2h')}",
            "--gpus-per-node=1",
            "--ntasks-per-node=8",
            "--array=111-119",
            "--output=logs/patch-tune-%A-%a.out",
            "--error=logs/patch-tune-%A-%a.err",
            script,
        ],
        dependency=f"afterok:{feat_id}",
    )
    print(f"  feat_tune_id={feat_tune_id}")

    # ── Step 4: Tuning aggregate ──────────────────────────────────────────────
    print("Submitting tuning-aggregate …")
    tuning_agg_id = sbatch(
        [
            "--export=ALL,HYDRA_JOB=tuning-aggregate",
            "--job-name=tcga-ut-tuning-agg",
            f"--partition={_env('TUNING_AGGREGATE_PARTITION', 'cpu-2h')}",
            "--gpus-per-node=0",
            "--ntasks-per-node=2",
            "--output=logs/tuning-aggregate-%j.out",
            "--error=logs/tuning-aggregate-%j.err",
            script,
        ],
        dependency=f"afterok:{feat_train_id}:{feat_tune_id}",
    )
    print(f"  tuning_agg_id={tuning_agg_id}")

    # ── Step 5a: Aggregate tables ─────────────────────────────────────────────
    print("Submitting aggregate …")
    agg_id = sbatch(
        [
            "--export=ALL,HYDRA_JOB=aggregate",
            "--job-name=tcga-ut-aggregate",
            "--partition=cpu-2h",
            "--gpus-per-node=0",
            "--ntasks-per-node=2",
            "--output=logs/aggregate-%j.out",
            "--error=logs/aggregate-%j.err",
            script,
        ],
        dependency=f"afterok:{tuning_agg_id}",
    )
    print(f"  agg_id={agg_id}")

    # ── Step 5b: ProGAN diagnostics ───────────────────────────────────────────
    print("Submitting progan-diagnostics …")
    diag_id = sbatch(
        [
            "--export=ALL,HYDRA_JOB=progan-diagnostics",
            "--job-name=tcga-ut-progan-diag",
            "--partition=cpu-2h",
            "--gpus-per-node=0",
            "--ntasks-per-node=4",
            "--output=logs/progan-diagnostics-%j.out",
            "--error=logs/progan-diagnostics-%j.err",
            script,
        ],
        dependency=f"afterok:{agg_id}",
    )
    print(f"  diag_id={diag_id}")

    print("\n=== Chain submitted ===")
    print(f"  sqfs:         {sqfs_id}")
    print(f"  feat-extract: {feat_id}")
    print(f"  feat-train:   {feat_train_id}")
    print(f"  feat-tune:    {feat_tune_id}")
    print(f"  tuning-agg:   {tuning_agg_id}")
    print(f"  aggregate:    {agg_id}")
    print(f"  progan-diag:  {diag_id}")
    print(
        "\nMonitor with: squeue -u $USER -o '%%i %%j %%T %%M %%R' | sort -k1"
    )


if __name__ == "__main__":
    main()
