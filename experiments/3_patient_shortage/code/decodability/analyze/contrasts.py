"""Compute bootstrap distributions, contrasts, and confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.gates import confidence_interval
from imbalance_benchmark.analysis.query import (
    load_seed_predictions,
    read_run_record,
)
from imbalance_benchmark.common import ensure_dirs, split_paths

from decodability import (
    BOOTSTRAP_SEED,
    MLP_ASSIGNMENT,
    N_REPLICATES,
    SUPPORTS,
    exp2_split_paths,
    probe_dir,
)

__all__ = ["compute_contrasts", "load_readout_predictions"]

_READOUTS = ("mlp", "logreg", "knn")
_PROBES = ("logreg", "knn")


def load_readout_predictions(
    config: dict[str, Any],
    split_index: int,
    support: str,
    readout: str,
    selection: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Load test labels and prediction stack for readout (stack shape: (S, N))."""
    exp2_paths = exp2_split_paths(config, split_index)
    if readout == "mlp":
        assignment = MLP_ASSIGNMENT[support]
        stacked = load_seed_predictions(
            exp2_paths, support, "ce", assignment=assignment, fields=("preds",)
        )
        if stacked is None:
            raise RuntimeError(f"Missing MLP predictions for {support}")
        return stacked["labels"], stacked["preds"]

    supp_sel = selection.get("supports", {}).get(support, {}).get(readout, {})
    param_str = supp_sel.get("selected_param_str")
    paths = split_paths(ensure_dirs(config), split_index)
    r_dir = probe_dir(paths, support, readout, param_str)
    record = read_run_record(r_dir, splits=("test",), array_fields=("labels", "preds"))
    if record is None or "test" not in record.get("splits", {}):
        raise RuntimeError(f"Missing test run record in {r_dir}")
    test_data = record["splits"]["test"]
    return np.asarray(test_data["labels"]), np.asarray(test_data["preds"])[
        np.newaxis, :
    ]


def _split_ba_distribution(
    config: dict[str, Any],
    split_index: int,
    support: str,
    readout: str,
    selection: dict[str, Any],
    n_classes: int,
) -> np.ndarray:
    """Build replicate balanced accuracy distribution for one split."""
    exp2_paths = exp2_split_paths(config, split_index)
    context = BootstrapContext(
        exp2_paths, is_mil=False, n_replicates=N_REPLICATES, seed=BOOTSTRAP_SEED
    )
    labels, preds_stack = load_readout_predictions(
        config, split_index, support, readout, selection
    )
    return context.ba_distribution(labels, preds_stack, n_classes)


def _collect_pooled_dists(
    config: dict[str, Any], selection: dict[str, Any], n_classes: int
) -> dict[str, dict[str, np.ndarray]]:
    """Collect 3-split pooled distributions for all supports and readouts."""
    dists: dict[str, dict[str, np.ndarray]] = {s: {} for s in SUPPORTS}
    for s in SUPPORTS:
        for r in _READOUTS:
            splits = [
                _split_ba_distribution(config, i, s, r, selection, n_classes)
                for i in range(3)
            ]
            if not all(np.all(np.isfinite(d)) for d in splits):
                raise RuntimeError(f"Non-finite bootstrap replicates for {s}/{r}")
            dists[s][r] = np.mean(splits, axis=0)
    return dists


def _pack_contrast(arr: np.ndarray) -> dict[str, Any]:
    """Scale by 100 and extract point estimate and 95% CI."""
    arr_pct = arr * 100.0
    ci = confidence_interval(arr_pct)
    return {"point": float(arr_pct[0]), "ci_2_5": float(ci[0]), "ci_97_5": float(ci[1])}


def _calculate_effects(
    dists: dict[str, dict[str, np.ndarray]],
) -> tuple[
    dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]
]:
    """Calculate coverage benefits, gains, and interaction distributions."""
    a_c, a_s = dists["balanced"], dists["balanced_spread"]
    b_benefit = {r: a_s[r] - a_c[r] for r in _READOUTS}
    g_gain = {
        "balanced": {r: a_c[r] - a_c["mlp"] for r in _PROBES},
        "balanced_spread": {r: a_s[r] - a_s["mlp"] for r in _PROBES},
    }
    i_interaction = {
        r: g_gain["balanced"][r] - g_gain["balanced_spread"][r] for r in _PROBES
    }
    for r in _PROBES:
        if not np.allclose(
            i_interaction[r], b_benefit["mlp"] - b_benefit[r], atol=1e-12
        ):
            raise RuntimeError(f"Interaction identity violation for {r}")
    return b_benefit, g_gain, i_interaction


def compute_contrasts(
    config: dict[str, Any],
    selection: dict[str, Any],
    n_classes: int,
) -> dict[str, Any]:
    """Compute pooled distributions, gains, coverage benefits, and interactions."""
    dists = _collect_pooled_dists(config, selection, n_classes)
    b_benefit, g_gain, i_interaction = _calculate_effects(dists)
    return {
        "pooled_accuracies": {
            s: {r: _pack_contrast(dists[s][r]) for r in _READOUTS} for s in SUPPORTS
        },
        "coverage_benefits": {r: _pack_contrast(b_benefit[r]) for r in _READOUTS},
        "classifier_gains": {
            s: {r: _pack_contrast(g_gain[s][r]) for r in _PROBES} for s in SUPPORTS
        },
        "interactions": {r: _pack_contrast(i_interaction[r]) for r in _PROBES},
    }
