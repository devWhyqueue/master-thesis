"""Hydra job definitions for the native BRACS benchmark."""

from __future__ import annotations

import argparse

from analysis.evaluation.native_tuning_grid import task_count as native_task_count
from bracs.common_jobs import (
    bracs_data_root,
    bracs_progan_cmd,
    bracs_root,
    mode_gate,
    native_tune_cmd,
    report_output_dir,
)
from job_defs import Job, prefix


def bracs_stage(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Verify BRACS source files and pre-convert BRACS.xlsx to CSV outside the container."""
    data_root = bracs_data_root(config)
    root = bracs_root(config)
    working_dir = config.get("working_dir", ".")
    metadata_csv = f"{root}/BRACS_metadata.csv"
    convert_script = f"{working_dir}/hydra/bracs/convert_xlsx.py"
    cmd = ["bash", "-lc", _stage_bash(data_root, convert_script, metadata_csv)]
    return [
        Job(
            cmd,
            "bracs_stage",
            "logs/bracs/bracs_stage%j.out",
            partition=config.get("bracs_stage_partition", "cpu-test"),
            cpus_per_task=1,
        )
    ]


def _stage_bash(data_root: str, convert_script: str, metadata_csv: str) -> str:
    """Build the bracs_stage bash command string."""
    return (
        f"(test -d {data_root}/BRACS_WSI && "
        f"test -d {data_root}/BRACS_RoI && "
        f"test -f {data_root}/BRACS.xlsx) || "
        f"(echo 'ERROR: BRACS source files missing under {data_root}' && exit 1); "
        f"python3 {convert_script} {data_root} {metadata_csv}; "
        f"echo 'BRACS staged: metadata CSV ready at {metadata_csv}'"
    )


def bracs_prepare(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Prepare BRACS ROI tiles and native manifests."""
    root = bracs_root(config)
    data_root = bracs_data_root(config)
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "data.bracs.prepare",
                f"--bracs-root={data_root}",
                f"--output-root={root}",
                f"--metadata-csv={root}/BRACS_metadata.csv",
                "--tile-size=256",
                "--max-tiles-per-roi=30",
                "--seeds",
                "0",
                "1",
                "2",
            ],
            "bracs_prepare",
            "logs/bracs/bracs_prepare%j.out",
            partition=config.get("bracs_prepare_partition", "cpu-2h"),
            cpus_per_task=int(config.get("cpus_per_task", "4")),
        )
    ]


def bracs_features(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Extract Virchow2 features for each BRACS native split seed."""
    return [_bracs_feature_job(args, config, seed) for seed in (0, 1, 2)]


def _bracs_feature_job(
    args: argparse.Namespace, config: dict[str, str], seed: int
) -> Job:
    root = bracs_root(config)
    stem = f"{root}/manifests/native_seed={seed}"
    return Job(
        prefix(config, args, gpu=True)
        + [
            "-m",
            "data.bracs.features",
            f"--manifest-dir={stem}",
            f"--feature-path={root}/features/virchow2/native_seed={seed}.pt",
            f"--batch-size={config.get('bracs_feature_batch_size', '64')}",
            f"--dtype={config.get('bracs_feature_dtype', 'float16')}",
            f"--device={config.get('bracs_feature_device', 'cuda')}",
        ],
        "bracs_features",
        "logs/bracs/bracs_features%j.out",
        partition=config.get("bracs_feature_partition", "gpu-2h"),
        gpus_per_node=1,
        cpus_per_task=int(config.get("cpus_per_task", "4")),
        constraint=config.get("bracs_gpu_constraint") or None,
    )


def bracs_patch_cache(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build row-level feature caches for native BRACS patch training."""
    root = bracs_root(config)
    return [
        Job(
            mode_gate(config, args, "native")
            + [
                "--",
                "hydra/build_feature_cache.py",
                f"--manifest-path={root}/manifests/native_seed={seed}/manifest_splits.csv",
                f"--file-save-path={root}/manifests/native_seed={seed}/patch_feature_cache.pt",
            ],
            "bracs_patch_cache",
            "logs/bracs/bracs_patch_cache%j.out",
            partition="cpu-9m",
        )
        for seed in (0, 1, 2)
    ]


def bracs_wsi_cache(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build BRACS native WSI bag caches."""
    root = bracs_root(config)
    return [
        Job(
            mode_gate(config, args, "native")
            + [
                "--",
                "-m",
                "modeling.training.constructed_wsi_cache",
                f"--manifest-path={root}/manifests/native_seed={seed}/manifest_splits.csv",
                f"--cache-dir={root}/manifests/native_seed={seed}/wsi_bag_cache",
            ],
            "bracs_wsi_cache",
            "logs/bracs/bracs_wsi_cache%j.out",
            partition="cpu-2h",
        )
        for seed in (0, 1, 2)
    ]


def bracs_progan_cache(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build BRACS native ProGAN augmented manifests and feature caches."""
    return [
        Job(
            bracs_progan_cmd(args, config, seed),
            "bracs_progan",
            "logs/bracs/bracs_progan%j.out",
            partition=config.get("progan_partition", "gpu-5h"),
            gpus_per_node=1,
            mem=config.get("progan_finalize_mem", "128G"),
            constraint=config.get("progan_gpu_constraint") or None,
        )
        for seed in (0, 1, 2)
    ]


def bracs_tune(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Submit BRACS native patch tuning array."""
    count = native_task_count("patch")
    return [
        Job(
            native_tune_cmd(args, config, "patch"),
            "bracs_tune_patch",
            "logs/bracs/bracs_tune_patch_%A_%a.out",
            partition=config.get("tune_partition", "cpu-2h"),
            array_spec=f"0-{count - 1}%{config.get('tune_array_throttle', '50')}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            mem=config.get("tune_mem", "32G"),
        )
    ]


def bracs_tune_wsi(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Submit BRACS native WSI tuning array."""
    count = native_task_count("wsi")
    return [
        Job(
            native_tune_cmd(args, config, "wsi", gpu=True),
            "bracs_tune_wsi",
            "logs/bracs/bracs_tune_wsi_%A_%a.out",
            partition=config.get("wsi_partition", "gpu-9m"),
            gpus_per_node=1,
            array_spec=f"0-{count - 1}%{config.get('wsi_tune_array_throttle', '20')}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
        )
    ]


def bracs_tune_aggregate(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Aggregate BRACS native tuning."""
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "analysis.evaluation.native_tuning_aggregate",
                f"--results-dir={config.get('bracs_results_dir', '')}",
                f"--output-dir={report_output_dir(config)}",
            ],
            "bracs_tune_agg",
            "logs/bracs/bracs_tune_agg%j.out",
        )
    ]


def bracs_report(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build BRACS native report artifacts."""
    root = bracs_root(config)
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "analysis.plotting.bracs_native_report",
                f"--manifest-root={root}/manifests",
                f"--prepare-report={root}/bracs_prepare_report.json",
                f"--results-dir={config.get('bracs_results_dir', '')}",
                f"--output-dir={report_output_dir(config)}",
            ],
            "bracs_report",
            "logs/bracs/bracs_report%j.out",
        )
    ]
