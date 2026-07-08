"""Hydra job definitions for the native CAMELYON16 benchmark."""

from __future__ import annotations

import argparse

from analysis.evaluation.native_tuning_grid import task_count as native_task_count
from camelyon16.common_jobs import (
    camelyon16_data_root,
    camelyon16_progan_cmd,
    camelyon16_root,
    native_tune_cmd,
    report_dir,
)
from job_defs import Job, prefix


def camelyon16_prepare(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Prepare CAMELYON16 native manifests from pre-tiled patches and masks."""
    root = camelyon16_root(config)
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "data.camelyon16.prepare",
                f"--data-root={camelyon16_data_root(config)}",
                f"--output-root={root}",
                "--seeds",
                "0",
                "1",
                "2",
            ],
            "cam16_prepare",
            "logs/camelyon16/cam16_prepare%j.out",
            partition=config.get("camelyon16_prepare_partition", "cpu-2h"),
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            mem=config.get("camelyon16_prepare_mem", "32G"),
        )
    ]


def camelyon16_features(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Extract per-slide Virchow2 features as a SLURM array over slides."""
    root = camelyon16_root(config)
    count = int(config.get("camelyon16_n_slides", "398"))
    return [
        Job(
            prefix(config, args, gpu=True)
            + [
                "-m",
                "data.camelyon16.features",
                f"--manifest-path={root}/manifests/native_seed=0/manifest_splits.csv",
                f"--feature-dir={root}/features/virchow2",
                f"--batch-size={config.get('camelyon16_feature_batch_size', '64')}",
                f"--dtype={config.get('camelyon16_feature_dtype', 'float16')}",
                f"--device={config.get('camelyon16_feature_device', 'cuda')}",
            ],
            "cam16_features",
            "logs/camelyon16/cam16_features_%A_%a.out",
            partition=config.get("camelyon16_feature_partition", "gpu-2h"),
            gpus_per_node=1,
            array_spec=f"0-{count - 1}%{config.get('camelyon16_feature_throttle', '32')}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            constraint=config.get("camelyon16_gpu_constraint") or None,
        )
    ]


def camelyon16_patch_cache(
    args: argparse.Namespace, config: dict[str, str]
) -> list[Job]:
    """Build row-level feature caches for the CAMELYON16 patch pool."""
    root = camelyon16_root(config)
    return [
        Job(
            prefix(config, args)
            + [
                "hydra/build_feature_cache.py",
                f"--manifest-path={root}/manifests/native_seed={seed}/patch_manifest.csv",
                f"--file-save-path={root}/manifests/native_seed={seed}/patch_feature_cache.pt",
            ],
            "cam16_patch_cache",
            "logs/camelyon16/cam16_patch_cache%j.out",
            partition=config.get("camelyon16_cache_partition", "cpu-2h"),
            mem=config.get("camelyon16_cache_mem", "32G"),
        )
        for seed in (0, 1, 2)
    ]


def camelyon16_progan_cache(
    args: argparse.Namespace, config: dict[str, str]
) -> list[Job]:
    """Build CAMELYON16 ProGAN augmented manifests and feature caches."""
    return [
        Job(
            camelyon16_progan_cmd(args, config, seed),
            "cam16_progan",
            "logs/camelyon16/cam16_progan%j.out",
            partition=config.get("progan_partition", "gpu-5h"),
            gpus_per_node=1,
            mem=config.get("progan_finalize_mem", "128G"),
            constraint=config.get("progan_gpu_constraint") or None,
        )
        for seed in (0, 1, 2)
    ]


def camelyon16_tune(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Submit CAMELYON16 native patch tuning array."""
    count = native_task_count("patch")
    return [
        Job(
            native_tune_cmd(args, config, "patch"),
            "cam16_tune_patch",
            "logs/camelyon16/cam16_tune_patch_%A_%a.out",
            partition=config.get("tune_partition", "cpu-2h"),
            array_spec=f"0-{count - 1}%{config.get('tune_array_throttle', '50')}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            mem=config.get("camelyon16_tune_mem", "48G"),
        )
    ]


def camelyon16_tune_wsi(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Submit CAMELYON16 native WSI tuning array.

    cpus_per_task is pinned to 1 (0 DataLoader workers) because
    ``load_slide_features`` caches whole per-slide feature tensors in a
    process-local ``lru_cache``. CAMELYON16's bags are ~80MB/slide (bag size
    ~7874 vs BRACS's ~30), so each forked DataLoader worker duplicating that
    cache multiplies host memory several-fold and can OOM even a generous
    --mem; a single worker-free process holds the whole split resident once.
    """
    count = native_task_count("wsi")
    return [
        Job(
            native_tune_cmd(args, config, "wsi", gpu=True),
            "cam16_tune_wsi",
            "logs/camelyon16/cam16_tune_wsi_%A_%a.out",
            partition=config.get("camelyon16_wsi_partition", "gpu-2h"),
            gpus_per_node=1,
            array_spec=f"0-{count - 1}%{config.get('wsi_tune_array_throttle', '20')}",
            cpus_per_task=int(config.get("camelyon16_wsi_cpus_per_task", "1")),
            mem=config.get("camelyon16_wsi_mem", "128G"),
            constraint=config.get("camelyon16_gpu_constraint") or None,
        )
    ]


def camelyon16_tune_aggregate(
    args: argparse.Namespace, config: dict[str, str]
) -> list[Job]:
    """Aggregate CAMELYON16 native tuning and select winners."""
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "analysis.evaluation.native_tuning_aggregate",
                f"--results-dir={config.get('camelyon16_results_dir', '')}",
                f"--output-dir={report_dir(config)}/camelyon16",
            ],
            "cam16_tune_agg",
            "logs/camelyon16/cam16_tune_agg%j.out",
        )
    ]


def camelyon16_report(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build CAMELYON16 native report artifacts."""
    root = camelyon16_root(config)
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "analysis.plotting.camelyon16_native_report",
                f"--prepare-report={root}/camelyon16_prepare_report.json",
                f"--results-dir={config.get('camelyon16_results_dir', '')}",
                f"--selection-dir={report_dir(config)}/camelyon16",
                f"--output-dir={report_dir(config)}",
            ],
            "cam16_report",
            "logs/camelyon16/cam16_report%j.out",
        )
    ]
