from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.predictors.rq3_cross_split import load_rq3_cells
from imbalance_benchmark.common import bag_dataset_kwargs
from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    fit_deficit_model,
    fit_gate_pass_model,
    fit_linked_sensitivity_models,
    fit_recovery_model,
    leave_one_group_out,
)
from imbalance_benchmark.analysis.predictors.separability import (
    class_margin_cross_fit,
    condition_learnability,
    effective_support,
    intrinsic_separability,
    intraclass_correlation,
)
from imbalance_benchmark.analysis.predictors.rq3_features import (
    feature_frame as _feature_frame,
    feature_identity as _feature_identity,
)

__all__ = ["run_rq3", "cross_dataset_rq3", "load_rq3_cells"]


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


def _covariates(
    paths: dict[str, Path],
    is_mil: bool,
    condition: dict[str, Any],
    freeze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Frozen-feature RQ3 covariates measured before mitigation fitting.

    ``separability`` (intrinsic, fixed balanced reference) is the second primary
    predictor; the rest (condition-specific learnability, balanced-kNN recall,
    per-class NN error, log minimum independent support, and the patch-vs-WSI
    indicator) are descriptive covariates for the single-predictor sensitivity
    fits only.
    """
    bag_kwargs = bag_dataset_kwargs({}, freeze) if is_mil else None
    class_names = list((freeze or {}).get("class_names", [])) or None
    ref_path = paths["data"] / "manifest_balanced.csv"
    ref_x, ref_y = _feature_frame(ref_path, None, is_mil, class_names, bag_kwargs)
    val_x, val_y = _feature_frame(
        paths["data"] / "manifest.csv", "validation", is_mil, class_names, bag_kwargs
    )
    n_classes = len(np.unique(ref_y))
    intrinsic = intrinsic_separability(ref_x, ref_y, val_x, val_y, n_classes)
    cond_path = Path(condition["path"])
    if not cond_path.exists():
        raise RuntimeError(f"Missing frozen controlled manifest for RQ3: {cond_path}")
    cond_x, cond_y = _feature_frame(cond_path, None, is_mil, class_names, bag_kwargs)
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
    reference_frame = _feature_identity(ref_path, None, is_mil, class_names, bag_kwargs)
    margins = class_margin_cross_fit(
        ref_x, ref_y, reference_frame["case_id"].astype(str).to_numpy(), n_classes
    )
    condition_frame = _feature_identity(
        cond_path, None, is_mil, class_names, bag_kwargs
    )
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
    paths: dict[str, Path],
    comparisons: list[dict[str, Any]],
    freeze: dict[str, Any],
    group: str,
    is_mil: bool,
) -> list[dict[str, Any]]:
    """Turn gate/recovery output into the three linked RQ3 observation sets."""
    gate_map = {(row["assignment"], row["severity"]): False for row in comparisons}
    for row in comparisons:
        if row["method"] == "ce":
            gate_map[(row["assignment"], row["severity"])] |= bool(row["gate_passed"])
    cells = []
    for row in comparisons:
        is_deficit_cell = row["method"] == "ce"
        if is_deficit_cell and row["gate"] != "discrimination":
            continue
        allocated = freeze["assignment_conditions"][row["assignment"]][row["severity"]]
        cells.append(
            {
                "group": group,
                "assignment": row["assignment"],
                "severity": row["severity"],
                "gate": row["gate"],
                "rho": allocated["achieved_rho"],
                "gate_passed": (
                    gate_map[(row["assignment"], row["severity"])]
                    if is_deficit_cell
                    else bool(row["gate_passed"])
                ),
                "deficit_ba": row["effect"] if is_deficit_cell else np.nan,
                "deficit_se": _standard_error(row),
                "recovery": row.get("recovery", np.nan),
                "recovery_se": _recovery_standard_error(row),
                "method": row["method"],
                **_covariates(paths, is_mil, allocated, freeze),
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
    is_mil = freeze.get("dataset_provenance", {}).get("regime") == "wsi"
    group = _rq3_group(freeze)
    cells = _cells(paths, comparisons, freeze, group, is_mil)
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


def _rq3_group(freeze: dict[str, Any]) -> str:
    """Return the frozen dataset-target random-intercept identifier."""
    dataset = freeze.get("dataset_provenance", {})
    target = dataset.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(
            "Frozen dataset.target is required to define an RQ3 dataset-target group"
        )
    return f"{dataset.get('name', 'unknown')}:{target}"


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
        "sensitivity": (
            fit_linked_sensitivity_models(deficit_cells, recovery_cells)
            if deficit_cells
            else {}
        ),
        "leave_one_group_out": leave_one_group_out(deficit_cells)
        if len(groups) > 1
        else {},
    }
