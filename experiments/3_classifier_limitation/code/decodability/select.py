"""Validation hyperparameter selection and signed lock."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.query import read_run_record
from imbalance_benchmark.common import (
    ensure_dirs,
    output_root,
    sign_file,
    split_paths,
    verify_signed_file,
)

from decodability import (
    K_VALUES,
    LAMBDAS,
    SUPPORTS,
    TIE_TOLERANCE,
    probe_dir,
)

__all__ = ["run_select"]

logger = logging.getLogger(__name__)


def _load_val_score(
    paths: dict[str, Path], support: str, readout: str, param: str
) -> float:
    """Load validation patient-macro balanced accuracy for one candidate."""
    r_dir = probe_dir(paths, support, readout, param)
    record = read_run_record(r_dir, splits=("validation",))
    if record is None or "validation" not in record.get("splits", {}):
        raise RuntimeError(f"Missing validation record in {r_dir}")
    endpoints = record["splits"]["validation"].get("endpoints", {})
    if "patient_macro_balanced_accuracy" not in endpoints:
        raise RuntimeError(
            f"Missing patient_macro_balanced_accuracy in {r_dir}/run.json"
        )
    return float(endpoints["patient_macro_balanced_accuracy"])


def _select_logreg(config: dict[str, Any], support: str) -> dict[str, Any]:
    """Select best lambda across splits: resolve <= 1e-10 ties toward larger lambda."""
    scores: dict[str, list[float]] = {}
    unavailable: list[float] = []

    for lam in LAMBDAS:
        param_str = f"lambda={lam}"
        split_scores = []
        is_avail = True
        for split_index in range(3):
            paths = split_paths(ensure_dirs(config), split_index)
            r_dir = probe_dir(paths, support, "logreg", param_str)
            rec = read_run_record(r_dir, splits=("validation",))
            if rec is None or not rec.get("solver", {}).get("converged", False):
                is_avail = False
                break
            split_scores.append(_load_val_score(paths, support, "logreg", param_str))
        if is_avail:
            scores[str(lam)] = split_scores
        else:
            unavailable.append(lam)

    if not scores:
        raise RuntimeError(f"No logreg candidate converged for {support}")

    mean_scores = {lam: float(np.mean(vals)) for lam, vals in scores.items()}
    # Sort lambda candidates ascending
    sorted_lams = sorted(mean_scores.keys(), key=lambda x: float(x))
    best_lam = sorted_lams[0]
    best_score = mean_scores[best_lam]

    for lam in sorted_lams[1:]:
        score = mean_scores[lam]
        diff = score - best_score
        # If score strictly exceeds or ties within TIE_TOLERANCE, prefer larger lambda
        if diff > TIE_TOLERANCE or abs(diff) <= TIE_TOLERANCE:
            best_lam = lam
            best_score = score

    best_lam_float = float(best_lam)
    is_boundary = best_lam_float in (LAMBDAS[0], LAMBDAS[-1])
    return {
        "selected": best_lam_float,
        "selected_param_str": f"lambda={best_lam_float}",
        "mean_score": best_score,
        "all_scores": mean_scores,
        "unavailable": unavailable,
        "is_boundary": is_boundary,
    }


def _select_knn(config: dict[str, Any], support: str) -> dict[str, Any]:
    """Select best k across splits: resolve <= 1e-10 ties toward larger k."""
    scores: dict[str, list[float]] = {}
    for k in K_VALUES:
        param_str = f"k={k}"
        split_scores = []
        for split_index in range(3):
            paths = split_paths(ensure_dirs(config), split_index)
            split_scores.append(_load_val_score(paths, support, "knn", param_str))
        scores[str(k)] = split_scores

    mean_scores = {k: float(np.mean(vals)) for k, vals in scores.items()}
    sorted_ks = sorted(mean_scores.keys(), key=lambda x: int(x))
    best_k = sorted_ks[0]
    best_score = mean_scores[best_k]

    for k in sorted_ks[1:]:
        score = mean_scores[k]
        diff = score - best_score
        # If score strictly exceeds or ties within TIE_TOLERANCE, prefer larger k
        if diff > TIE_TOLERANCE or abs(diff) <= TIE_TOLERANCE:
            best_k = k
            best_score = score

    best_k_int = int(best_k)
    is_boundary = best_k_int in (K_VALUES[0], K_VALUES[-1])
    return {
        "selected": best_k_int,
        "selected_param_str": f"k={best_k_int}",
        "mean_score": best_score,
        "all_scores": mean_scores,
        "unavailable": [],
        "is_boundary": is_boundary,
    }


def run_select(config: dict[str, Any]) -> Path:
    """Select hyperparameters for logreg and knn for each support condition."""
    root_p = output_root(config)
    preflight_p = root_p / "data" / "preflight.json"
    verify_signed_file(preflight_p)

    selection: dict[str, Any] = {"supports": {}}
    for support in SUPPORTS:
        logreg_sel = _select_logreg(config, support)
        knn_sel = _select_knn(config, support)
        selection["supports"][support] = {
            "logreg": logreg_sel,
            "knn": knn_sel,
        }

    out_p = root_p / "data" / "probe_selection.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    sign_file(out_p)
    logger.info("Probe selection signed at %s", out_p)
    return out_p
