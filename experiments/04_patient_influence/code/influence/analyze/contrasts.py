"""Compute bootstrap distributions and the W/B/I contrasts (report eqs. 8-10)."""

from __future__ import annotations

from typing import Any

import numpy as np
from decodability import exp2_split_paths
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.gates import confidence_interval
from imbalance_benchmark.analysis.query import read_run_record
from imbalance_benchmark.common import ensure_dirs, split_paths

from influence import BOOTSTRAP_SEED, N_REPLICATES, OBJECTIVES, SUPPORTS, cell_dir
from influence.baseline import load_baseline_predictions

__all__ = ["compute_contrasts", "load_objective_predictions"]


def _load_own_predictions(
    config: dict[str, Any], split_index: int, support: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load exp-4's own patient-average test predictions for one cell."""
    paths = split_paths(ensure_dirs(config), split_index)
    record = read_run_record(
        cell_dir(paths, support, "patient"),
        splits=("test",),
        array_fields=("labels", "preds"),
    )
    if record is None or "test" not in record.get("splits", {}):
        raise RuntimeError(
            f"Missing patient-average record for split {split_index}, support {support}"
        )
    test_data = record["splits"]["test"]
    return np.asarray(test_data["labels"]), np.asarray(test_data["preds"])


def load_objective_predictions(
    config: dict[str, Any], split_index: int, support: str, objective: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load test labels and a single-row prediction stack for one objective."""
    if objective == "patch":
        labels, preds = load_baseline_predictions(config, split_index, support)
    elif objective == "patient":
        labels, preds = _load_own_predictions(config, split_index, support)
    else:
        raise ValueError(f"Unknown objective: {objective!r}")
    return labels, preds[np.newaxis, :]


def _collect_pooled_dists(
    config: dict[str, Any], n_classes: int
) -> dict[str, dict[str, np.ndarray]]:
    """Collect 3-split pooled replicate distributions for all supports and objectives."""
    contexts = {
        i: BootstrapContext(
            exp2_split_paths(config, i),
            is_mil=False,
            n_replicates=N_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        for i in range(3)
    }
    dists: dict[str, dict[str, np.ndarray]] = {s: {} for s in SUPPORTS}
    for s in SUPPORTS:
        for o in OBJECTIVES:
            splits = [
                contexts[i].ba_distribution(
                    *load_objective_predictions(config, i, s, o), n_classes
                )
                for i in range(3)
            ]
            if not all(np.all(np.isfinite(d)) for d in splits):
                raise RuntimeError(f"Non-finite bootstrap replicates for {s}/{o}")
            dists[s][o] = np.mean(splits, axis=0)
    return dists


def _pack_contrast(arr: np.ndarray) -> dict[str, Any]:
    """Scale by 100 and extract point estimate and 95% CI."""
    arr_pct = arr * 100.0
    ci = confidence_interval(arr_pct)
    return {"point": float(arr_pct[0]), "ci_2_5": float(ci[0]), "ci_97_5": float(ci[1])}


def compute_contrasts(config: dict[str, Any], n_classes: int) -> dict[str, Any]:
    """Compute pooled accuracies, weighting gains W_s, coverage benefits B_o, and I."""
    dists = _collect_pooled_dists(config, n_classes)
    w_gain = {s: dists[s]["patient"] - dists[s]["patch"] for s in SUPPORTS}
    b_benefit = {
        o: dists["balanced_spread"][o] - dists["balanced"][o] for o in OBJECTIVES
    }
    interaction = w_gain["balanced"] - w_gain["balanced_spread"]
    if not np.allclose(
        interaction, b_benefit["patch"] - b_benefit["patient"], atol=1e-12
    ):
        raise RuntimeError(
            "Interaction identity violation: W_C - W_S != B_patch - B_patient"
        )

    return {
        "pooled_accuracies": {
            s: {o: _pack_contrast(dists[s][o]) for o in OBJECTIVES} for s in SUPPORTS
        },
        "weighting_gains": {s: _pack_contrast(w_gain[s]) for s in SUPPORTS},
        "coverage_benefits": {o: _pack_contrast(b_benefit[o]) for o in OBJECTIVES},
        "interaction": _pack_contrast(interaction),
    }
