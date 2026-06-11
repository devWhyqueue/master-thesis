"""Run one validation-tuning task for patch or WSI constructed training."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from common_code.tuning.registry import patch_feature_method_flags
from tcga_ut_imbalanced.evaluation.tuning_grid import task_for_index

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse tuning runner CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["patch", "wsi"])
    parser.add_argument("--array-task-id", type=int, default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--constructed-dataset-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run one tuning task or skip it when outputs already exist."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    task_index = _resolve_array_task_id(args)
    task = task_for_index(args.benchmark, task_index)
    out = _output_dir(args, task)
    if _output_complete(out):
        logger.info("Skipping completed task: %s", out)
        return
    cmd = (
        _patch_command(args, task, out)
        if args.benchmark == "patch"
        else _wsi_command(args, task, out)
    )
    if args.dry_run:
        logger.info(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def _resolve_array_task_id(args: argparse.Namespace) -> int:
    if args.array_task_id is not None:
        return args.array_task_id
    env_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_value is not None:
        return int(env_value)
    raise ValueError(
        "Missing task index: pass --array-task-id or run inside a SLURM array job."
    )


def _output_dir(args: argparse.Namespace, task) -> str:
    benchmark = "patch" if args.benchmark == "patch" else "wsi"
    return (
        f"{args.results_dir}/tuning/{benchmark}/"
        f"{task.regime.label}/{task.variant.method}/{task.variant.variant}/"
        f"seed={task.seed}"
    )


def _output_complete(out: str) -> bool:
    out_path = Path(out)
    return (out_path / "validation_results.json").exists() and (
        out_path / "test_results.json"
    ).exists()


def _patch_command(args: argparse.Namespace, task, out: str) -> list[str]:
    stem = _constructed_stem(
        args.constructed_dataset_dir,
        task.regime.class_order_name,
        task.regime.parameter,
        task.seed,
    )
    manifest = f"{stem}/manifest_splits.csv"
    cmd = [
        sys.executable,
        "-m",
        "tcga_ut_imbalanced.cli.train",
        f"--dataset-structure-path={manifest}",
        "--dataset-split=train",
        f"--validation-dataset-structure-path={manifest}",
        "--validation-dataset-split=validation",
        f"--test-dataset-structure-path={manifest}",
        "--test-dataset-split=test",
        f"--feature-path={args.feature_path}",
        "--preload-features",
        f"--results-save-path={out}",
        "--model=mlp",
        f"--training-method={task.variant.method}",
        f"--tuning-id={task.variant.variant}",
        f"--tuning-params={task.variant.params_json}",
        f"--seed={task.seed}",
        "--device=cpu",
        "--optimizer=adamw",
        "--learning-rate=0.001",
        "--weight-decay=0.0001",
        "--n-epochs=30",
        "--batch-size=256",
        "--dropout=0.1",
        f"--class-names-path={stem}/class_order.json",
    ]
    cmd.extend(patch_feature_method_flags(task.variant.method))
    cache_path = Path(stem) / "patch_feature_cache.pt"
    if cache_path.is_file():
        cmd.append(f"--feature-cache-path={cache_path}")
    return cmd


def _wsi_command(args: argparse.Namespace, task, out: str) -> list[str]:
    stem = _constructed_stem(
        args.constructed_dataset_dir,
        task.regime.class_order_name,
        task.regime.parameter,
        task.seed,
    )
    cmd = [
        sys.executable,
        "-m",
        "tcga_ut_imbalanced.training.constructed_wsi",
        f"--manifest-path={stem}/manifest_splits.csv",
        f"--results-save-path={out}",
        f"--method={task.variant.method}",
        f"--seed={task.seed}",
        f"--class-order-name={task.regime.class_order_name}",
        f"--parameter={task.regime.parameter}",
        f"--tuning-id={task.variant.variant}",
        f"--tuning-params={task.variant.params_json}",
        "--device=auto",
        "--epochs=30",
        "--bag-batch-size=32",
        "--max-instances-per-bag=30",
        f"--bag-cache-dir={stem}/wsi_bag_cache",
    ]
    cmd.extend(_wsi_tuning_flags(task.variant.params))
    return cmd


def _wsi_tuning_flags(params: dict[str, float]) -> list[str]:
    mapping = {
        "weight_power": "--weight-power",
        "focal_gamma": "--focal-gamma",
        "sampler_power": "--sampler-power",
        "rankmix_alpha": "--rankmix-alpha",
        "sc_mil_temperature": "--sc-mil-temperature",
        "mde_mil_consistency_weight": "--mde-mil-consistency-weight",
    }
    return [f"{flag}={params[key]:g}" for key, flag in mapping.items() if key in params]


def _constructed_stem(
    root: str, class_order_name: str, parameter: float, seed: int
) -> str:
    name = f"constructed_order={class_order_name}_parameter={parameter}_seed={seed}"
    return f"{root}/{name}"


if __name__ == "__main__":
    main()
