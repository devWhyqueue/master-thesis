from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    fit_deficit_model,
    fit_gate_pass_model,
    fit_recovery_model,
    fit_sensitivity_models,
    leave_one_group_out,
)
from imbalance_benchmark.analysis.predictors.separability import (
    condition_learnability,
    intrinsic_separability,
)
from imbalance_benchmark.datasets.data import BagFeatureDataset, ImbalanceDataset

__all__ = ["run_rq3", "cross_dataset_rq3"]


def _feature_frame(
    manifest: Path, split: str | None, is_mil: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Load fixed embeddings and integer targets from one frozen manifest partition."""
    if is_mil:
        dataset = BagFeatureDataset(manifest, split)
        features = []
        for index in range(len(dataset)):
            bag, _ = dataset[index]
            features.append(np.r_[bag.mean(0).cpu(), bag.std(0).cpu()])
    else:
        dataset = ImbalanceDataset(manifest, split)
        features = [
            dataset[index]["features"].cpu().numpy() for index in range(len(dataset))
        ]
    return np.asarray(features), dataset.get_int_targets()


def _min_independent_support(freeze: dict[str, Any]) -> float:
    """Smallest realized per-class allocated support across the balanced condition."""
    allocated = (
        freeze.get("conditions", {}).get("balanced", {}).get("allocated_counts", {})
    )
    values = [v for v in allocated.values() if v]
    return float(min(values)) if values else 1.0


def _covariates(
    paths: dict[str, Path], is_mil: bool, freeze: dict[str, Any]
) -> dict[str, Any]:
    """Frozen-feature RQ3 covariates measured before mitigation fitting.

    ``separability`` (intrinsic, fixed balanced reference) is the second primary
    predictor; the rest (condition-specific learnability, balanced-kNN recall,
    per-class NN error, log minimum independent support, and the patch-vs-WSI
    indicator) are descriptive covariates for the single-predictor sensitivity
    fits only.
    """
    ref_x, ref_y = _feature_frame(paths["data"] / "manifest_balanced.csv", None, is_mil)
    val_x, val_y = _feature_frame(paths["data"] / "manifest.csv", "validation", is_mil)
    n_classes = len(np.unique(ref_y))
    intrinsic = intrinsic_separability(ref_x, ref_y, val_x, val_y, n_classes)
    severe = paths["data"] / "manifest_severe.csv"
    cond_path = severe if severe.exists() else paths["data"] / "manifest_balanced.csv"
    cond_x, cond_y = _feature_frame(cond_path, None, is_mil)
    learnability = condition_learnability(cond_x, cond_y, val_x, val_y, n_classes)
    return {
        "separability": float(intrinsic["linear_probe_macro_recall"]),
        "knn_macro_recall": float(intrinsic["knn_macro_recall"]),
        "per_class_nn_error": intrinsic["per_class_nn_error"],
        "learnability": float(learnability["linear_probe_macro_recall"]),
        "log_min_support": float(np.log(_min_independent_support(freeze))),
        "is_wsi": 1.0 if is_mil else 0.0,
    }


def _standard_error(comparison: dict[str, Any]) -> float:
    """Bootstrap standard error of the raw effect from its stored replicate distribution."""
    return float(np.nanstd(np.asarray(comparison["bootstrap_effect"]), ddof=1))


def _recovery_standard_error(comparison: dict[str, Any]) -> float:
    """Bootstrap standard error of the recovery ratio."""
    numerator = np.asarray(comparison.get("bootstrap_numerator", []), dtype=float)
    denominator = np.asarray(comparison.get("bootstrap_denominator", []), dtype=float)
    if numerator.size == 0 or denominator.size != numerator.size:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        recovery = np.where(denominator != 0, numerator / denominator, np.nan)
    return float(np.nanstd(recovery, ddof=1))


def _cells(
    comparisons: list[dict[str, Any]],
    freeze: dict[str, Any],
    group: str,
    covariates: dict[str, Any],
) -> list[dict[str, Any]]:
    """Turn gate/recovery output into the three linked RQ3 observation sets."""
    gate_map = {(row["assignment"], row["severity"]): False for row in comparisons}
    for row in comparisons:
        if row["method"] == "ce":
            gate_map[(row["assignment"], row["severity"])] |= bool(row["gate_passed"])
    cells = []
    for row in comparisons:
        if row["gate"] != "discrimination":
            continue
        allocated = freeze["assignment_conditions"][row["assignment"]][row["severity"]]
        cells.append(
            {
                "group": group,
                "rho": allocated["achieved_rho"],
                "gate_passed": gate_map[(row["assignment"], row["severity"])],
                "deficit_ba": row["effect"] if row["method"] == "ce" else np.nan,
                "deficit_se": _standard_error(row),
                "recovery": row.get("recovery", np.nan),
                "recovery_se": _recovery_standard_error(row),
                "method": row["method"],
                **covariates,
            }
        )
    return cells


def run_rq3(
    paths: dict[str, Path],
    config: dict[str, Any],
    freeze: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute frozen-feature RQ3 cells, fits, and descriptive covariate records.

    A single dataset-regime yields only one random-intercept group, so its
    within-split fits are degenerate; the cross-dataset combination is the
    inferential unit (see :func:`cross_dataset_rq3`). This still emits the cells
    and covariates each split contributes to that combined analysis.
    """
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    group = f"{config.get('dataset', {}).get('name', 'unknown')}:{'wsi' if is_mil else 'patch'}"
    cells = _cells(comparisons, freeze, group, _covariates(paths, is_mil, freeze))
    deficit_cells = [cell for cell in cells if cell["method"] == "ce"]
    recovery_cells = [cell for cell in cells if cell["method"] != "ce"]
    return {
        "cells": cells,
        "models": {
            "gate_pass": fit_gate_pass_model(deficit_cells) if deficit_cells else {},
            "deficit": fit_deficit_model(deficit_cells) if deficit_cells else {},
            "recovery": fit_recovery_model(recovery_cells),
        },
    }


def cross_dataset_rq3(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Combined RQ3 across dataset-target groups: the actual inferential unit."""
    deficit_cells = [c for c in cells if c["method"] == "ce"]
    recovery_cells = [c for c in cells if c["method"] != "ce"]
    groups = sorted({c["group"] for c in cells})
    return {
        "n_groups": len(groups),
        "groups": groups,
        "cells": cells,
        "models": {
            "gate_pass": fit_gate_pass_model(deficit_cells) if deficit_cells else {},
            "deficit": fit_deficit_model(deficit_cells) if deficit_cells else {},
            "recovery": fit_recovery_model(recovery_cells),
        },
        "sensitivity": fit_sensitivity_models(deficit_cells) if deficit_cells else {},
        "leave_one_group_out": leave_one_group_out(deficit_cells)
        if len(groups) > 1
        else {},
    }


def load_rq3_cells(analysis_roots: list[Path]) -> list[dict[str, Any]]:
    """Gather every analyzed dataset-regime's RQ3 cells for the combined fit."""
    cells: list[dict[str, Any]] = []
    for root in analysis_roots:
        for index in range(3):
            rq3_path = root / f"split={index}" / "data" / "rq3.json"
            if rq3_path.exists():
                cells.extend(json.loads(rq3_path.read_text()).get("cells", []))
    return cells
