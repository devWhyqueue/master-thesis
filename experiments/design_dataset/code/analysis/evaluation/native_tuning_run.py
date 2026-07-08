"""Run one native-dataset validation-tuning task."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from common_code.tuning.registry import patch_feature_method_flags
from analysis.evaluation.native_tuning_grid import task_for_index

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse native tuning runner arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["patch", "wsi"])
    parser.add_argument("--array-task-id", type=int, default=None)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--prepare-report", default=None)
    parser.add_argument("--required-mode", default=None)
    parser.add_argument("--max-instances-per-bag", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run one native tuning task."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if _skip_for_mode(args.prepare_report, args.required_mode):
        return
    task = task_for_index(args.benchmark, _resolve_array_task_id(args))
    out = _output_dir(args.results_dir, args.benchmark, task.variant, task.seed)
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
    if env_value is None:
        raise ValueError("Pass --array-task-id or run in a SLURM array.")
    return int(env_value)


def _skip_for_mode(report_path: str | None, required_mode: str | None) -> bool:
    if report_path is None or required_mode is None:
        return False
    path = Path(report_path)
    if not path.exists():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    mode = str(report.get("recommended_benchmark_mode", "native"))
    if mode == required_mode:
        return False
    logger.info("Skipping because BRACS mode is %s, required %s.", mode, required_mode)
    return True


def _output_dir(results_dir: str, benchmark: str, variant, seed: int) -> str:
    return (
        f"{results_dir}/tuning/{benchmark}/native/{variant.method}/"
        f"{variant.variant}/seed={seed}"
    )


def _output_complete(out: str) -> bool:
    path = Path(out)
    return (path / "validation_results.json").exists() and (
        path / "test_results.json"
    ).exists()


def _seed_dir(manifest_root: str, seed: int) -> Path:
    return Path(manifest_root) / f"native_seed={seed}"


def _patch_command(args: argparse.Namespace, task, out: str) -> list[str]:
    stem = _seed_dir(args.manifest_root, task.seed)
    train = stem / "train.csv"
    validation = stem / "validation.csv"
    test = stem / "test.csv"
    cache = stem / "patch_feature_cache.pt"
    dataset_split = None
    validation_split = None
    test_split = None
    if task.variant.method == "patch_feature_progan_aug":
        train = stem / "manifest_splits_progan.csv"
        validation = train
        test = train
        cache = stem / "patch_feature_cache_progan.pt"
        dataset_split = "train"
        validation_split = "validation"
        test_split = "test"
    cmd = [
        sys.executable,
        "-m",
        "cli.train",
        f"--dataset-structure-path={train}",
        f"--validation-dataset-structure-path={validation}",
        f"--test-dataset-structure-path={test}",
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
        f"--class-names-path={stem / 'class_order.json'}",
    ]
    if dataset_split is not None:
        cmd.append(f"--dataset-split={dataset_split}")
        cmd.append(f"--validation-dataset-split={validation_split}")
        cmd.append(f"--test-dataset-split={test_split}")
    if cache.is_file():
        cmd.append(f"--feature-cache-path={cache}")
    cmd.extend(patch_feature_method_flags(task.variant.method))
    return cmd


def _wsi_command(args: argparse.Namespace, task, out: str) -> list[str]:
    stem = _seed_dir(args.manifest_root, task.seed)
    cmd = [
        sys.executable,
        "-m",
        "modeling.training.constructed_wsi",
        f"--manifest-path={stem / 'manifest_splits.csv'}",
        f"--results-save-path={out}",
        f"--method={task.variant.method}",
        f"--seed={task.seed}",
        "--class-order-name=native",
        "--parameter=0.0",
        f"--tuning-id={task.variant.variant}",
        f"--tuning-params={task.variant.params_json}",
        "--device=auto",
        "--epochs=30",
        "--bag-batch-size=32",
        f"--max-instances-per-bag={args.max_instances_per_bag}",
        f"--bag-cache-dir={stem / 'wsi_bag_cache'}",
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


if __name__ == "__main__":
    main()
