"""Build BRACS power-law benchmark manifests when native imbalance is too weak."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import cast

import pandas as pd

from data.full_scale.rows import slide_frame
from data.full_scale.sampling import (
    available_training_counts,
    class_order,
    constructed_payload,
    load_manifest,
    max_feasible_total,
    output_dir_for_args,
    split_frames,
    write_constructed_outputs,
)

LAMBDAS = (0.5, 1.0, 1.5)
SEEDS = (0, 1, 2)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse BRACS power-law construction arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bracs-root", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build BRACS power-law manifests when requested by preparation report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    root = Path(args.bracs_root)
    report = _load_report(root / "bracs_prepare_report.json")
    if report.get("recommended_benchmark_mode") != "power_law" and not args.force:
        logger.info(
            "BRACS native distribution is imbalanced enough; skipping power-law."
        )
        return
    output_root = root / "constructed_power_law"
    pool_size = _pool_size(root)
    for parameter in LAMBDAS:
        for seed in SEEDS:
            _write_constructed(root, output_root, parameter, seed, pool_size)
    _write_power_law_report(root, output_root, pool_size)


def _pool_size(root: Path) -> int:
    totals = []
    for parameter in LAMBDAS:
        for seed in SEEDS:
            manifest = load_manifest(str(_native_manifest(root, seed)))
            ordered = class_order(manifest, None)
            splits = split_frames(_args(parameter, seed, root), manifest)
            available = cast(
                pd.Series,
                slide_frame(splits["train"]).groupby("cancer_type")["slide_id"].nunique(),
            )
            totals.append(max_feasible_total(available, ordered, parameter))
    pool = min(totals)
    logger.info("BRACS power-law pool size: %s", pool)
    return int(pool)


def _write_constructed(
    root: Path, output_root: Path, parameter: float, seed: int, pool_size: int
) -> None:
    args = _args(parameter, seed, root, output_root, pool_size)
    manifest = load_manifest(str(_native_manifest(root, seed)))
    ordered = class_order(manifest, None)
    splits = split_frames(args, manifest)
    frames, targets = constructed_payload(args, splits, ordered)
    output_dir = output_dir_for_args(args)
    write_constructed_outputs(
        frames,
        targets,
        ordered,
        output_dir,
        vars(args),
        feature_dir=None,
        available_counts=available_training_counts(splits["train"]),
    )
    logger.info("Wrote %s", output_dir)


def _args(
    parameter: float,
    seed: int,
    root: Path,
    output_root: Path | None = None,
    pool_size: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        slide_manifest_path=str(_native_manifest(root, seed)),
        split_assignment_path=None,
        split_column="split",
        train_name="train",
        validation_name="validation",
        test_name="test",
        parameter=parameter,
        seed=seed,
        pool_size=pool_size,
        n_patches_per_slide=30,
        file_save_path=str(output_root or root / "constructed_power_law"),
        class_order_name="native_prevalence",
        class_order_file=None,
    )


def _native_manifest(root: Path, seed: int) -> Path:
    return root / "manifests" / f"native_seed={seed}" / "manifest_splits.csv"


def _load_report(path: Path) -> dict:
    if not path.exists():
        return {"recommended_benchmark_mode": "power_law"}
    return cast(dict, json.loads(path.read_text(encoding="utf-8")))


def _write_power_law_report(root: Path, output_root: Path, pool_size: int) -> None:
    payload = {
        "dataset": "BRACS",
        "benchmark_mode": "power_law",
        "lambdas": list(LAMBDAS),
        "seeds": list(SEEDS),
        "pool_size": pool_size,
        "constructed_dataset_dir": str(output_root),
    }
    (root / "bracs_power_law_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
