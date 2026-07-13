from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    fit_deficit_model,
    fit_gate_pass_model,
    fit_recovery_model,
)
from imbalance_benchmark.analysis.predictors.separability import intrinsic_separability
from imbalance_benchmark.datasets.data import BagFeatureDataset, ImbalanceDataset

__all__ = ["run_rq3"]


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


def _separability(paths: dict[str, Path], is_mil: bool) -> float:
    """Measure the fixed balanced-reference linear probe on natural validation data."""
    ref_x, ref_y = _feature_frame(paths["data"] / "manifest_balanced.csv", None, is_mil)
    val_x, val_y = _feature_frame(paths["data"] / "manifest.csv", "validation", is_mil)
    return float(
        intrinsic_separability(ref_x, ref_y, val_x, val_y, len(np.unique(ref_y)))[
            "linear_probe_macro_recall"
        ]
    )


def _standard_error(comparison: dict[str, Any]) -> float:
    """Estimate a bootstrap standard error from the stored replicate distribution."""
    return float(np.nanstd(np.asarray(comparison["bootstrap_effect"]), ddof=1))


def _cells(
    comparisons: list[dict[str, Any]],
    freeze: dict[str, Any],
    group: str,
    separability: float,
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
                "separability": separability,
                "gate_passed": gate_map[(row["assignment"], row["severity"])],
                "deficit_ba": row["effect"] if row["method"] == "ce" else np.nan,
                "deficit_se": _standard_error(row),
                "recovery": row.get("recovery", np.nan),
                "recovery_se": _standard_error(row),
                "method": row["method"],
            }
        )
    return cells


def run_rq3(
    paths: dict[str, Path],
    config: dict[str, Any],
    freeze: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute frozen-feature RQ3 cells, fits, and descriptive covariate records."""
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    group = f"{config.get('dataset', {}).get('name', 'unknown')}:{'wsi' if is_mil else 'patch'}"
    cells = _cells(comparisons, freeze, group, _separability(paths, is_mil))
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
