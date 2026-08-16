from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.predictors.separability import (
    class_margin_cross_fit,
    condition_learnability,
    effective_support,
    intrinsic_separability,
    intraclass_correlation,
)
from imbalance_benchmark.datasets.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    load_training_dataset,
)
from imbalance_benchmark.datasets.features.cache import bank_index


def feature_frame(
    manifest: Path,
    split: str | None,
    is_mil: bool,
    class_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load fixed embeddings and integer targets from one frozen manifest partition."""
    dataset = load_training_dataset(manifest, is_mil, split, class_names=class_names)
    if is_mil:
        bags = cast(BagFeatureDataset, dataset)
        features = [np.r_[bag.mean(0).cpu(), bag.std(0).cpu()] for bag, _ in bags]
    else:
        patches = cast(ImbalanceDataset, dataset)
        features = bank_index(patches.rows).cpu().numpy()
    return np.asarray(features), dataset.get_int_targets()


def feature_identity(
    manifest: Path,
    split: str | None,
    is_mil: bool,
    class_names: list[str] | None,
) -> pd.DataFrame:
    """Return identities in the same one-row-per-observation order as features."""
    dataset = load_training_dataset(manifest, is_mil, split, class_names=class_names)
    return cast(pd.DataFrame, dataset.df[["case_id", "slide_id"]]).reset_index(
        drop=True
    )


def _min_support(condition: dict[str, Any], key: str) -> float:
    """Return the smallest class-specific support for one manifest statistic."""
    values = [stats[key] for stats in condition["contribution_stats"].values()]
    return float(min(values)) if values else 1.0


def _min_independent_support(condition: dict[str, Any], is_mil: bool) -> float:
    """Return the smallest contributing-patient/slide support in one condition."""
    return _min_support(condition, "n_slides" if is_mil else "n_patients")


def _has_multiple_slides_per_patient(condition: dict[str, Any]) -> bool:
    """Whether any class's WSI condition includes repeat patient contributions."""
    return any(
        stats["n_slides"] > stats["n_patients"]
        for stats in condition["contribution_stats"].values()
    )


def _reference_block(
    paths: dict[str, Path],
    is_mil: bool,
    class_names: list[str] | None,
) -> dict[str, Any]:
    """Compute the per-split, condition-invariant reference/validation covariate inputs once.

    ``ref_x``/``ref_y``/``val_x``/``val_y``/``n_classes``/``intrinsic`` depend only on
    ``manifest_balanced.csv`` and the validation partition, not on the per-condition
    manifest. The patch-regime margin/ICC cross-fit is likewise reference-only, so it is
    computed here too and reused across every ``(assignment, severity)`` cell.
    """
    ref_path = paths["data"] / "manifest_balanced.csv"
    # Seed the resident bank at full-manifest capacity before loading subsets.
    val_x, val_y = feature_frame(
        paths["data"] / "manifest.csv", "validation", is_mil, class_names
    )
    ref_x, ref_y = feature_frame(ref_path, None, is_mil, class_names)
    n_classes = len(np.unique(ref_y))
    intrinsic = intrinsic_separability(ref_x, ref_y, val_x, val_y, n_classes)
    block = {
        "ref_x": ref_x,
        "ref_y": ref_y,
        "val_x": val_x,
        "val_y": val_y,
        "n_classes": n_classes,
        "intrinsic": intrinsic,
    }
    if is_mil:
        return block
    reference_frame = feature_identity(ref_path, None, is_mil, class_names)
    reference_cases = reference_frame["case_id"].astype(str).to_numpy()
    margins = class_margin_cross_fit(ref_x, ref_y, reference_cases, n_classes)
    block["reference_frame"] = reference_frame
    block["reference_cases"] = reference_cases
    block["margins"] = margins
    return block


def _covariates(
    paths: dict[str, Path],
    is_mil: bool,
    condition: dict[str, Any],
    reference: dict[str, Any],
    freeze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Frozen-feature RQ3 covariates measured before mitigation fitting."""
    class_names = list((freeze or {}).get("class_names", [])) or None
    ref_x, ref_y = reference["ref_x"], reference["ref_y"]
    val_x, val_y = reference["val_x"], reference["val_y"]
    n_classes = reference["n_classes"]
    intrinsic = reference["intrinsic"]
    cond_path = Path(condition["path"])
    if not cond_path.exists():
        raise RuntimeError(f"Missing frozen controlled manifest for RQ3: {cond_path}")
    cond_x, cond_y = feature_frame(cond_path, None, is_mil, class_names)
    learnability = condition_learnability(cond_x, cond_y, val_x, val_y, n_classes)
    covariates = {
        "separability": float(intrinsic["linear_probe_macro_recall"]),
        "knn_macro_recall": float(intrinsic["knn_macro_recall"]),
        "per_class_nn_error": intrinsic["per_class_nn_error"],
        "learnability": float(learnability["linear_probe_macro_recall"]),
        "log_min_support": float(np.log(_min_independent_support(condition, is_mil))),
        "is_wsi": 1.0 if is_mil else 0.0,
    }
    if is_mil:
        if _has_multiple_slides_per_patient(condition):
            covariates["log_min_patient_support"] = float(
                np.log(_min_support(condition, "n_patients"))
            )
        return covariates
    reference_frame = reference["reference_frame"]
    margins = reference["margins"]
    condition_frame = feature_identity(cond_path, None, is_mil, class_names)
    effective = []
    for class_index in range(n_classes):
        reference_mask = ref_y == class_index
        condition_mask = cond_y == class_index
        reference_cases = (
            reference_frame.loc[reference_mask, "case_id"].astype(str).to_numpy()
        )
        condition_cases = (
            condition_frame.loc[condition_mask, "case_id"].astype(str).to_numpy()
        )
        counts = pd.Series(condition_cases).value_counts()
        effective.append(
            effective_support(
                int(condition_mask.sum()),
                float(counts.mean()),
                intraclass_correlation(margins[reference_mask], reference_cases),
            )
        )
    covariates["log_effective_support"] = float(np.log(max(1.0, min(effective))))
    return covariates
