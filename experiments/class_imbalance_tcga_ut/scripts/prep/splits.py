from __future__ import annotations

import argparse
import logging
from typing import cast

import numpy as np
import pandas as pd

from scripts.common import ensure_dirs, load_config, write_json

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for split generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def assign_class_splits(
    slides: list[str], seed: int, val_fraction: float, test_fraction: float
) -> dict[str, str]:
    """Assign per-class slide IDs to train/val/test."""
    rng = np.random.default_rng(seed)
    shuffled = list(slides)
    rng.shuffle(shuffled)
    n_slides = len(shuffled)
    if n_slides == 1:
        return {shuffled[0]: "train"}
    if n_slides == 2:
        return {shuffled[0]: "train", shuffled[1]: "test"}

    n_test = max(1, int(round(n_slides * test_fraction)))
    n_val = max(1, int(round(n_slides * val_fraction)))
    if n_test + n_val >= n_slides:
        n_test = 1
        n_val = 1

    assignments = {}
    for slide_id in shuffled[:n_test]:
        assignments[slide_id] = "test"
    for slide_id in shuffled[n_test : n_test + n_val]:
        assignments[slide_id] = "val"
    for slide_id in shuffled[n_test + n_val :]:
        assignments[slide_id] = "train"
    return assignments


def _build_assignments(
    slide_manifest: pd.DataFrame, seed: int, data_config: dict
) -> dict[str, str]:
    """Build slide-level split assignments class by class."""
    assignments: dict[str, str] = {}
    for _, class_df in slide_manifest.groupby("cancer_type"):
        class_assignments = assign_class_splits(
            class_df["slide_id"].tolist(),
            seed=seed,
            val_fraction=float(data_config["validation_fraction"]),
            test_fraction=float(data_config["test_fraction"]),
        )
        assignments.update(class_assignments)
    return assignments


def _validate_assignments(split_manifest: pd.DataFrame) -> None:
    """Fail fast if any rows did not receive a split."""
    if not bool(split_manifest["split"].isna().any()):
        return
    missing = split_manifest.loc[split_manifest["split"].isna(), "slide_id"].unique()[
        :10
    ]
    raise RuntimeError(f"Missing split assignments for slides: {missing}")


def _add_balanced_test_flag(
    split_manifest: pd.DataFrame,
    slide_splits: pd.DataFrame,
    balanced_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Mark rows selected for optional balanced test subset."""
    if balanced_per_class <= 0:
        split_manifest["balanced_test"] = False
        return split_manifest
    rng = np.random.default_rng(seed)
    balanced_ids: list[str] = []
    test_frame = slide_splits[slide_splits["split"] == "test"]
    for _, class_df in test_frame.groupby("cancer_type"):
        ids = np.asarray(class_df["slide_id"])
        take = min(balanced_per_class, len(ids))
        if take:
            balanced_ids.extend(rng.choice(ids, size=take, replace=False).tolist())
    split_manifest["balanced_test"] = split_manifest["slide_id"].isin(balanced_ids)
    return split_manifest


def _write_split_outputs(
    paths: dict,
    seed: int,
    split_manifest: pd.DataFrame,
    slide_splits: pd.DataFrame,
) -> None:
    """Write split manifests, counts, and summary report."""
    split_path = paths["data"] / f"manifest_splits_seed={seed}.csv"
    slide_split_path = paths["data"] / f"slide_splits_seed={seed}.csv"
    split_manifest.to_csv(split_path, index=False)
    slide_splits.to_csv(slide_split_path, index=False)
    split_counts = cast(
        pd.DataFrame,
        slide_splits.groupby(["split", "cancer_type"])
        .size()
        .to_frame("n_slides")
        .reset_index(),
    )
    split_counts.to_csv(paths["data"] / f"split_counts_seed={seed}.csv", index=False)
    write_json(
        paths["data"] / f"split_report_seed={seed}.json",
        {
            "seed": seed,
            "n_slides_by_split": slide_splits["split"].value_counts().to_dict(),
            "n_rows_by_split": split_manifest["split"].value_counts().to_dict(),
            "balanced_test_rows": int(split_manifest["balanced_test"].sum()),
        },
    )
    logger.info(f"Wrote {split_path}")
    logger.info(f"Wrote {slide_split_path}")


def main() -> None:
    """Create reproducible train/validation/test splits."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    data_config = config["data"]
    manifest = pd.read_csv(paths["data"] / "manifest.csv")
    slide_manifest = pd.read_csv(paths["data"] / "slide_manifest.csv")
    assignments = _build_assignments(slide_manifest, args.seed, data_config)
    split_manifest = manifest.copy()
    split_manifest["split"] = split_manifest["slide_id"].map(assignments)
    _validate_assignments(split_manifest)
    slide_splits = slide_manifest.copy()
    slide_splits["split"] = slide_splits["slide_id"].map(assignments)
    balanced_per_class = int(data_config.get("balanced_test_per_class") or 0)
    split_manifest = _add_balanced_test_flag(
        split_manifest, slide_splits, balanced_per_class, args.seed
    )
    _write_split_outputs(paths, args.seed, split_manifest, slide_splits)


if __name__ == "__main__":
    main()
