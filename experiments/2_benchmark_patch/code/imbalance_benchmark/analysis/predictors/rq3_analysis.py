from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.predictors.rq3_cross_split import load_rq3_cells
from imbalance_benchmark.analysis.predictors.rq3_features import (
    _covariates,
    _reference_block,
)
from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    fit_deficit_model,
    fit_recovery_model,
    leave_one_group_out,
    support_difficulty_alignment,
)

__all__ = ["run_rq3", "cross_dataset_rq3", "load_rq3_cells"]

logger = logging.getLogger(__name__)


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
    """Turn gate/recovery output into RQ3 damage and recovery cells."""
    gate_map = {(row["assignment"], row["severity"]): False for row in comparisons}
    for row in comparisons:
        if row["method"] == "ce":
            gate_map[(row["assignment"], row["severity"])] |= bool(row["gate_passed"])
    cells = []
    class_names = list(freeze.get("class_names", [])) or None
    balanced = freeze["conditions"]["balanced"]
    reference = _reference_block(
        paths,
        is_mil,
        class_names,
        balanced,
        int(freeze["construction_seed"]),
    )
    # Shortages depend on the condition only, so each method reuses one calculation.
    covariate_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for row in comparisons:
        is_deficit_cell = row["method"] == "ce"
        if is_deficit_cell and row["gate"] != "discrimination":
            continue
        key = (row["assignment"], row["severity"])
        allocated = freeze["assignment_conditions"][row["assignment"]][row["severity"]]
        if key not in covariate_cache:
            logger.info("rq3: covariates %s/%s", *key)
            covariate_cache[key] = _covariates(
                paths, is_mil, allocated, reference, freeze
            )
        cells.append(
            {
                "group": group,
                "assignment": row["assignment"],
                "severity": row["severity"],
                "gate": row["gate"],
                "rho": allocated["achieved_rho"],
                "support_difficulty_alignment": support_difficulty_alignment(
                    allocated, freeze
                ),
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
                **covariate_cache[key],
            }
        )
    return cells


def run_rq3(
    paths: dict[str, Path],
    config: dict[str, Any],
    freeze: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute RQ3 signal-shortage cells and the two association fits.

    One dataset-regime has only one random-intercept group, so within-split fits
    are degenerate; the cross-dataset combination (:func:`cross_dataset_rq3`) is
    the actual inferential unit -- this only emits each split's contribution.
    """
    del config
    is_mil = freeze.get("dataset_provenance", {}).get("regime") == "wsi"
    group = _rq3_group(freeze)
    cells = _cells(paths, comparisons, freeze, group, is_mil)
    deficit_cells = [cell for cell in cells if cell["method"] == "ce"]
    recovery_cells = [cell for cell in cells if cell["method"] != "ce"]
    logger.info("rq3: %d cells ready, fitting damage model", len(cells))
    damage = fit_deficit_model(deficit_cells) if deficit_cells else {}
    logger.info("rq3: fitting recovery model")
    recovery = fit_recovery_model(recovery_cells)
    models = {"damage": damage, "recovery": recovery}
    return {"cells": cells, "models": models}


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
            "damage": fit_deficit_model(deficit_cells) if deficit_cells else {},
            "recovery": fit_recovery_model(recovery_cells),
        },
        "leave_one_group_out": leave_one_group_out(deficit_cells)
        if len(groups) > 1
        else {},
    }
