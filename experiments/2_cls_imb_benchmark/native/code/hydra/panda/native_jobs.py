"""Hydra job definitions for the native PANDA benchmark."""

from __future__ import annotations

import argparse

from analysis.evaluation.native_tuning_grid import task_count as native_task_count
from job_defs import Job, prefix
from panda.common_jobs import (
    native_tune_cmd,
    panda_data_root,
    panda_progan_cmd,
    panda_root,
    report_dir,
)


def _n_slides(config: dict[str, str]) -> int:
    return int(config.get("panda_n_slides", "2000"))


def panda_select(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Select the stratified PANDA slide subset shared by tiling and features."""
    root = panda_root(config)
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "data.panda.select",
                f"--data-root={panda_data_root(config)}",
                f"--output-root={root}",
                f"--n-slides={_n_slides(config)}",
                f"--seed={config.get('panda_select_seed', '0')}",
            ],
            "panda_select",
            "logs/panda/panda_select%j.out",
            partition=config.get("panda_prepare_partition", "cpu-2h"),
            mem=config.get("panda_prepare_mem", "16G"),
        )
    ]


def panda_tile(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Tile selected PANDA slides into 20x tissue tiles as a SLURM array."""
    root = panda_root(config)
    count = _n_slides(config)
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "data.panda.tiling",
                f"--selection-path={root}/selected_slides.csv",
                f"--tile-root={root}/patches/20x",
                f"--manifest-dir={root}/tiles",
                f"--tile-size={config.get('panda_tile_size', '256')}",
                f"--max-tiles={config.get('panda_max_tiles', '1000')}",
            ],
            "panda_tile",
            "logs/panda/panda_tile_%A_%a.out",
            partition=config.get("panda_tile_partition", "cpu-2h"),
            array_spec=f"0-{count - 1}%{config.get('panda_tile_throttle', '32')}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            mem=config.get("panda_tile_mem", "48G"),
        )
    ]


def panda_prepare(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Prepare PANDA native manifests from tiled patches."""
    root = panda_root(config)
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "data.panda.prepare",
                f"--output-root={root}",
                f"--selection-path={root}/selected_slides.csv",
                f"--tiles-dir={root}/tiles",
                "--seeds",
                "0",
                "1",
                "2",
            ],
            "panda_prepare",
            "logs/panda/panda_prepare%j.out",
            partition=config.get("panda_prepare_partition", "cpu-2h"),
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            mem=config.get("panda_prepare_mem", "32G"),
        )
    ]


def panda_features(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Extract per-slide Virchow2 features as a SLURM array over slides."""
    root = panda_root(config)
    count = _n_slides(config)
    return [
        Job(
            prefix(config, args, gpu=True)
            + [
                "-m",
                "data.panda.features",
                f"--manifest-path={root}/manifests/native_seed=0/manifest_splits.csv",
                f"--feature-dir={root}/features/virchow2",
                f"--batch-size={config.get('panda_feature_batch_size', '64')}",
                f"--dtype={config.get('panda_feature_dtype', 'float16')}",
                f"--device={config.get('panda_feature_device', 'cuda')}",
            ],
            "panda_features",
            "logs/panda/panda_features_%A_%a.out",
            partition=config.get("panda_feature_partition", "gpu-2h"),
            gpus_per_node=1,
            array_spec=f"0-{count - 1}%{config.get('panda_feature_throttle', '32')}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            constraint=config.get("panda_gpu_constraint") or None,
        )
    ]


def panda_patch_cache(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build row-level feature caches for the PANDA patch pool."""
    root = panda_root(config)
    return [
        Job(
            prefix(config, args)
            + [
                "hydra/build_feature_cache.py",
                f"--manifest-path={root}/manifests/native_seed={seed}/patch_manifest.csv",
                f"--file-save-path={root}/manifests/native_seed={seed}/patch_feature_cache.pt",
            ],
            "panda_patch_cache",
            "logs/panda/panda_patch_cache%j.out",
            partition=config.get("panda_cache_partition", "cpu-2h"),
            mem=config.get("panda_cache_mem", "32G"),
        )
        for seed in (0, 1, 2)
    ]


def panda_progan_cache(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build PANDA ProGAN augmented manifests and feature caches (one per seed)."""
    return [
        Job(
            panda_progan_cmd(args, config, seed),
            "panda_progan",
            "logs/panda/panda_progan%j.out",
            partition=config.get("progan_partition", "gpu-5h"),
            gpus_per_node=1,
            mem=config.get("progan_finalize_mem", "128G"),
            constraint=config.get("progan_gpu_constraint") or None,
        )
        for seed in (0, 1, 2)
    ]


def panda_tune(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Submit PANDA native patch tuning array."""
    count = native_task_count("patch")
    return [
        Job(
            native_tune_cmd(args, config, "patch"),
            "panda_tune_patch",
            "logs/panda/panda_tune_patch_%A_%a.out",
            partition=config.get("tune_partition", "cpu-2h"),
            array_spec=f"0-{count - 1}%{config.get('tune_array_throttle', '50')}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            mem=config.get("panda_tune_mem", "48G"),
        )
    ]


def panda_tune_wsi(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Submit PANDA native WSI tuning array."""
    count = native_task_count("wsi")
    return [
        Job(
            native_tune_cmd(args, config, "wsi", gpu=True),
            "panda_tune_wsi",
            "logs/panda/panda_tune_wsi_%A_%a.out",
            partition=config.get("panda_wsi_partition", "gpu-2h"),
            gpus_per_node=1,
            array_spec=f"0-{count - 1}%{config.get('wsi_tune_array_throttle', '20')}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            mem=config.get("panda_wsi_mem", "64G"),
            constraint=config.get("panda_gpu_constraint") or None,
        )
    ]


def panda_tune_aggregate(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Aggregate PANDA native tuning and select winners."""
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "analysis.evaluation.native_tuning_aggregate",
                f"--results-dir={config.get('panda_results_dir', '')}",
                f"--output-dir={report_dir(config)}/panda",
                "--allow-incomplete",
            ],
            "panda_tune_agg",
            "logs/panda/panda_tune_agg%j.out",
        )
    ]


def panda_report(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build PANDA native report artifacts."""
    root = panda_root(config)
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "analysis.plotting.panda_native_report",
                f"--prepare-report={root}/panda_prepare_report.json",
                f"--results-dir={config.get('panda_results_dir', '')}",
                f"--selection-dir={report_dir(config)}/panda",
                f"--output-dir={report_dir(config)}",
            ],
            "panda_report",
            "logs/panda/panda_report%j.out",
        )
    ]
