from __future__ import annotations

import argparse
import logging
from typing import cast

import numpy as np
import pandas as pd

from scripts.common import ensure_dirs, load_config, write_json
from scripts.data.prep.manifest.feature import tcga_case_id

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for split generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def assign_class_splits(
    units: list[str], seed: int, val_fraction: float, test_fraction: float
) -> dict[str, str]:
    """Assign per-class split units to train/val/test."""
    rng = np.random.default_rng(seed)
    shuffled = list(units)
    rng.shuffle(shuffled)
    n_units = len(shuffled)
    if n_units == 1:
        return {shuffled[0]: "train"}
    if n_units == 2:
        return {shuffled[0]: "train", shuffled[1]: "test"}

    n_test = max(1, int(round(n_units * test_fraction)))
    n_val = max(1, int(round(n_units * val_fraction)))
    if n_test + n_val >= n_units:
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


def _with_case_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing TCGA participant identifiers."""
    with_cases = frame.copy()
    if "case_id" not in with_cases.columns:
        with_cases["case_id"] = with_cases["slide_id"].map(
            lambda slide_id: tcga_case_id(str(slide_id))
        )
    return with_cases


def _build_assignments(
    slide_manifest: pd.DataFrame, seed: int, data_config: dict
) -> dict[str, str]:
    """Build case-level split assignments class by class."""
    assignments: dict[str, str] = {}
    manifest = _with_case_ids(slide_manifest)
    for _, class_df in manifest.groupby("cancer_type"):
        case_assignments = assign_class_splits(
            sorted(class_df["case_id"].astype(str).unique()),
            seed=seed,
            val_fraction=float(data_config["validation_fraction"]),
            test_fraction=float(data_config["test_fraction"]),
        )
        for _, slide in class_df.iterrows():
            assignments[str(slide["slide_id"])] = case_assignments[
                str(slide["case_id"])
            ]
    return assignments


def _validate_assignments(split_manifest: pd.DataFrame) -> None:
    """Fail fast if split assignments are missing or case-leaking."""
    if bool(split_manifest["split"].isna().any()):
        missing = split_manifest.loc[
            split_manifest["split"].isna(), "slide_id"
        ].unique()[:10]
        raise RuntimeError(f"Missing split assignments for slides: {missing}")
    case_split_counts = split_manifest.groupby("case_id")["split"].nunique()
    leaking_cases = cast(pd.Series, case_split_counts[case_split_counts > 1])
    if not leaking_cases.empty:
        examples = leaking_cases.index.astype(str).tolist()[:10]
        raise RuntimeError(f"Case-disjoint split violation for cases: {examples}")


def _map_slide_assignments(
    frame: pd.DataFrame, assignments: dict[str, str]
) -> pd.Series:
    """Map slide IDs to split names while preserving missing assignments as NaN."""
    split_series = frame["slide_id"].apply(
        lambda slide_id: assignments.get(str(slide_id))
    )
    return cast(pd.Series, split_series)


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
            "split_unit": "case_id",
            "n_slides_by_split": slide_splits["split"].value_counts().to_dict(),
            "n_cases_by_split": slide_splits.groupby("split")["case_id"]
            .nunique()
            .to_dict(),
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
    manifest = _with_case_ids(manifest)
    slide_manifest = _with_case_ids(slide_manifest)
    assignments = _build_assignments(slide_manifest, args.seed, data_config)
    split_manifest = manifest.copy()
    split_manifest["split"] = _map_slide_assignments(split_manifest, assignments)
    _validate_assignments(split_manifest)
    slide_splits = slide_manifest.copy()
    slide_splits["split"] = _map_slide_assignments(slide_splits, assignments)
    balanced_per_class = int(data_config.get("balanced_test_per_class") or 0)
    split_manifest = _add_balanced_test_flag(
        split_manifest, slide_splits, balanced_per_class, args.seed
    )
    _write_split_outputs(paths, args.seed, split_manifest, slide_splits)


if __name__ == "__main__":
    main()
