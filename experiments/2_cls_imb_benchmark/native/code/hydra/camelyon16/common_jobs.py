"""Shared command builders for CAMELYON16 Hydra jobs."""

from __future__ import annotations

import argparse

from bracs.common_jobs import progan_options, report_output_dir
from job_defs import prefix


def camelyon16_root(config: dict[str, str]) -> str:
    """Return the writable CAMELYON16 working root for derived artifacts."""
    return config.get("camelyon16_root", "/home/space/datasets/patho_ds/camelyon16")


def camelyon16_data_root(config: dict[str, str]) -> str:
    """Return the read-only CAMELYON16 source root (pre-existing shared dataset)."""
    return config.get("camelyon16_data_root", "/home/space/datasets/camelyon16")


def bag_size(config: dict[str, str]) -> str:
    """Return the WSI-bag instance budget (median available tiles per slide)."""
    return str(config.get("camelyon16_bag_size", "7874"))


def native_tune_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    benchmark: str,
    *,
    gpu: bool = False,
) -> list[str]:
    """Build a native CAMELYON16 tuning command."""
    root = camelyon16_root(config)
    command = prefix(config, args, gpu=gpu) + [
        "-m",
        "analysis.evaluation.native_tuning_run",
        f"--benchmark={benchmark}",
        f"--manifest-root={root}/manifests",
        f"--results-dir={config.get('camelyon16_results_dir', '')}",
        f"--feature-path={root}/features/virchow2",
        f"--prepare-report={root}/camelyon16_prepare_report.json",
        "--required-mode=native",
    ]
    if benchmark == "wsi":
        command.append(f"--max-instances-per-bag={bag_size(config)}")
    return command


def camelyon16_progan_cmd(
    args: argparse.Namespace, config: dict[str, str], seed: int
) -> list[str]:
    """Build the native CAMELYON16 ProGAN cache command over the patch manifest."""
    root = camelyon16_root(config)
    stem = f"{root}/manifests/native_seed={seed}"
    return prefix(config, args, gpu=True) + [
        "-m",
        "data.progan.cache",
        f"--manifest-path={stem}/patch_manifest.csv",
        f"--manifest-save-path={stem}/manifest_splits_progan.csv",
        f"--file-save-path={stem}/patch_feature_cache_progan.pt",
        f"--synthetic-root={stem}/synthetic_patch_images",
        f"--raw-root={camelyon16_data_root(config)}/patches/20x",
        "--raw-resolution=.",
        f"--seed={seed}",
        *progan_options(config),
    ]


def report_dir(config: dict[str, str]) -> str:
    """Return the design-dataset report output directory."""
    return report_output_dir(config)
