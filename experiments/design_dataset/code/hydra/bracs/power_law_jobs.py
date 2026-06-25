"""Hydra job definitions for the BRACS power-law fallback."""

from __future__ import annotations

import argparse

from analysis.evaluation.tuning_grid import task_count as constructed_task_count
from bracs.common_jobs import (
    bracs_root,
    constructed_tune_cmd,
    power_law_progan_cmd,
    power_law_report_dir,
)
from job_defs import Job, prefix


def bracs_power_law(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build BRACS power-law manifests if native imbalance is too weak."""
    root = bracs_root(config)
    return [
        Job(
            prefix(config, args)
            + ["-m", "data.bracs.power_law", f"--bracs-root={root}"],
            "bracs_power_law",
            "logs/bracs/bracs_power_law%j.out",
            partition="cpu-2h",
        )
    ]


def bracs_progan_power_law(
    args: argparse.Namespace, config: dict[str, str]
) -> list[Job]:
    """Build BRACS power-law ProGAN augmented manifests and feature caches."""
    jobs = []
    for parameter in (0.5, 1.0, 1.5):
        for seed in (0, 1, 2):
            jobs.append(
                Job(
                    power_law_progan_cmd(args, config, parameter, seed),
                    "bracs_pw_progan",
                    "logs/bracs/bracs_pw_progan%j.out",
                    partition=config.get("progan_partition", "gpu-5h"),
                    gpus_per_node=1,
                    mem=config.get("progan_finalize_mem", "128G"),
                    constraint=config.get("progan_gpu_constraint") or None,
                )
            )
    return jobs


def bracs_tune_power_law(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Submit BRACS power-law patch tuning array."""
    count = constructed_task_count("patch")
    return [
        Job(
            constructed_tune_cmd(args, config, "patch"),
            "bracs_pw_tune_patch",
            "logs/bracs/bracs_pw_tune_patch_%A_%a.out",
            partition=config.get("tune_partition", "cpu-2h"),
            array_spec=f"0-{count - 1}%{config.get('tune_array_throttle', '50')}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            mem=config.get("tune_mem", "32G"),
        )
    ]


def bracs_tune_wsi_power_law(
    args: argparse.Namespace, config: dict[str, str]
) -> list[Job]:
    """Submit BRACS power-law WSI tuning array."""
    count = constructed_task_count("wsi")
    return [
        Job(
            constructed_tune_cmd(args, config, "wsi", gpu=True),
            "bracs_pw_tune_wsi",
            "logs/bracs/bracs_pw_tune_wsi_%A_%a.out",
            partition=config.get("wsi_partition", "gpu-9m"),
            gpus_per_node=1,
            array_spec=f"0-{count - 1}%{config.get('wsi_tune_array_throttle', '20')}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
        )
    ]


def bracs_tune_aggregate_power_law(
    args: argparse.Namespace, config: dict[str, str]
) -> list[Job]:
    """Aggregate BRACS power-law tuning."""
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "analysis.evaluation.tuning_aggregate",
                f"--results-dir={config.get('bracs_results_dir', '')}",
                f"--output-dir={power_law_report_dir(config)}",
            ],
            "bracs_pw_tune_agg",
            "logs/bracs/bracs_pw_tune_agg%j.out",
        )
    ]


def bracs_report_power_law(
    args: argparse.Namespace, config: dict[str, str]
) -> list[Job]:
    """Build BRACS power-law report artifacts."""
    root = bracs_root(config)
    return [
        Job(
            prefix(config, args)
            + [
                "-m",
                "analysis.plotting.report",
                f"--constructed-dataset-dir={root}/constructed_power_law",
                f"--results-dir={config.get('bracs_results_dir', '')}",
                f"--output-dir={power_law_report_dir(config)}",
                "--no-native-reference",
            ],
            "bracs_pw_report",
            "logs/bracs/bracs_pw_report%j.out",
        )
    ]
