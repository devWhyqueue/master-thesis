"""Run one validation-tuning task for patch or WSI constructed training."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

from tcga_ut_imbalanced.evaluation.tuning_grid import task_for_index

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse validation-tuning runner arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["patch", "wsi"])
    parser.add_argument("--array-task-id", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--constructed-dataset-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--class-imbalance-root", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Dispatch one tuning task to the patch or WSI trainer."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    task = task_for_index(args.benchmark, args.array_task_id)
    cmd = (
        _patch_command(args, task)
        if args.benchmark == "patch"
        else _wsi_command(args, task)
    )
    if args.dry_run:
        logger.info(" ".join(cmd))
        return
    env = os.environ.copy()
    if args.class_imbalance_root:
        env["CLASS_IMBALANCE_ROOT"] = args.class_imbalance_root
    subprocess.run(cmd, check=True, env=env)


def _patch_command(args: argparse.Namespace, task) -> list[str]:
    stem = _constructed_stem(
        args.constructed_dataset_dir,
        task.regime.class_order_name,
        task.regime.parameter,
        task.seed,
    )
    out = (
        f"{args.results_dir}/tuning/patch/"
        f"{task.regime.label}/{task.variant.method}/{task.variant.variant}/"
        f"seed={task.seed}"
    )
    cmd = [
        sys.executable,
        "-m",
        "tcga_ut_imbalanced.cli.train",
        f"--dataset-structure-path={stem}/train.csv",
        f"--validation-dataset-structure-path={stem}/validation.csv",
        f"--test-dataset-structure-path={stem}/test.csv",
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
    cmd.extend(_patch_method_flags(task.variant.method))
    return cmd


def _wsi_command(args: argparse.Namespace, task) -> list[str]:
    stem = _constructed_stem(
        args.constructed_dataset_dir,
        task.regime.class_order_name,
        task.regime.parameter,
        task.seed,
    )
    out = (
        f"{args.results_dir}/tuning/wsi/"
        f"{task.regime.label}/{task.variant.method}/{task.variant.variant}/"
        f"seed={task.seed}"
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


def _patch_method_flags(method: str) -> list[str]:
    flags = {
        "ce": ["--loss=cross_entropy", "--alpha=uniform"],
        "weighted_ce": ["--loss=cross_entropy", "--alpha=inverse_class_frequency"],
        "balanced_sampler": [
            "--loss=cross_entropy",
            "--alpha=uniform",
            "--batch-balancing",
        ],
        "focal": ["--loss=focal_loss", "--alpha=uniform", "--gamma=2.0"],
        "ce_soft_f1": [
            "--loss=ce_soft_f1",
            "--alpha=uniform",
            "--batch-balancing",
        ],
        "ce_soft_mcc": [
            "--loss=ce_soft_mcc",
            "--alpha=uniform",
            "--batch-balancing",
        ],
    }
    return flags.get(method, [])


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
