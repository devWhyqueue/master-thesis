"""Pooled BA/NLL/ECE per encoder/arm and the prior/support/encoder decomposition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext

from breadth.analyze.secondary import (
    _patient_macro_recalls,
    _probability_quality,
    pack_estimate,
)
from breadth.calibrate import scaled_test_probabilities

from centre.analyze import pooled

from transfer import ENCODERS, RATIOS_NEW
from transfer.analyze.common import fit_dirs, require_record

__all__ = [
    "pooled_arms",
    "decomposition",
    "bonferroni_interval",
    "pack",
]

# Per-tail alpha for a two-sided 97.5% CI (Bonferroni over 2 primary contrasts, family >= 95%):
# total alpha 0.025 split into a 1.25% lower and 1.25% upper tail -> percentiles [1.25, 98.75].
_PRIMARY_ALPHA = 0.0125


def bonferroni_interval(dist: np.ndarray) -> tuple[float, float]:
    """Two-sided 97.5% percentile CI (index 0 is the observed replicate)."""
    replicates = dist[1:] if len(dist) > 1 else dist
    lo, hi = 100 * _PRIMARY_ALPHA, 100 * (1 - _PRIMARY_ALPHA)
    return (
        float(np.nanpercentile(replicates, lo)),
        float(np.nanpercentile(replicates, hi)),
    )


def pack(dist: np.ndarray, primary: bool = False) -> dict[str, float]:
    """Point estimate and 95% interval; add the primary contrast's 97.5% Bonferroni bounds too."""
    est = pack_estimate(dist)
    if primary:
        lo, hi = bonferroni_interval(dist)
        est["ci_1_25"], est["ci_98_75"] = lo, hi
    return est


def _ba_stack(
    paths: dict[int, dict[str, Path]],
    ctxs: dict[int, BootstrapContext],
    encoder: str,
    arm: str,
    n_classes: int,
) -> np.ndarray:
    """(F, R) patient-macro BA over every (split, draw) fit of one encoder's arm."""
    rows = []
    for s, _, result_dir in fit_dirs(paths, encoder, arm):
        rec = require_record(
            result_dir, splits=("test",), array_fields=("labels", "preds")
        )
        test = rec["splits"]["test"]
        labels, preds = np.asarray(test["labels"]), np.asarray(test["preds"])
        recalls = _patient_macro_recalls(ctxs[s], labels, preds, n_classes)
        rows.append(recalls.mean(axis=0) * 100.0)
    return np.stack(rows)


def _prob_stack(
    paths: dict[int, dict[str, Path]],
    ctxs: dict[int, BootstrapContext],
    encoder: str,
    arm: str,
    n_classes: int,
) -> dict[str, np.ndarray]:
    """(F, R) raw/TS macro-NLL/ECE over every (split, draw) fit of one encoder's arm."""
    cols: dict[str, list[np.ndarray]] = {
        "nll": [],
        "ece": [],
        "nll_ts": [],
        "ece_ts": [],
    }
    for s, _, result_dir in fit_dirs(paths, encoder, arm):
        rec = require_record(
            result_dir, splits=("test",), array_fields=("labels", "probabilities")
        )
        test = rec["splits"]["test"]
        labels, probs = np.asarray(test["labels"]), np.asarray(test["probabilities"])
        scaled = scaled_test_probabilities(result_dir, probs)
        nll, ece = _probability_quality(ctxs[s], labels, probs, n_classes)
        nll_ts, ece_ts = _probability_quality(ctxs[s], labels, scaled, n_classes)
        cols["nll"].append(nll)
        cols["ece"].append(ece)
        cols["nll_ts"].append(nll_ts)
        cols["ece_ts"].append(ece_ts)
    return {key: np.stack(vals) for key, vals in cols.items()}


def pooled_arms(
    paths: dict[int, dict[str, Path]],
    ctxs: dict[int, BootstrapContext],
    n_classes: int,
    arms: tuple[str, ...],
    prob_arms: tuple[str, ...],
    w: np.ndarray,
) -> tuple[
    dict[str, dict[str, np.ndarray]], dict[str, dict[str, dict[str, np.ndarray]]]
]:
    """Draw-weighted pooled BA per encoder/arm and NLL/ECE per encoder/probability-arm."""
    ba = {
        m: {arm: pooled(_ba_stack(paths, ctxs, m, arm, n_classes), w) for arm in arms}
        for m in ENCODERS
    }
    prob = {
        m: {
            arm: {
                k: pooled(d, w)
                for k, d in _prob_stack(paths, ctxs, m, arm, n_classes).items()
            }
            for arm in prob_arms
        }
        for m in ENCODERS
    }
    return ba, prob


def decomposition(ba: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """D_X(m,rho), I(m,rho), and delta_X(rho) = D_X(uni2h) - D_X(virchow2), for X in R/P/S."""
    dists: dict[str, np.ndarray] = {}
    for m in ENCODERS:
        for r in RATIOS_NEW:
            d_r, d_p, d_s = (ba[m]["B"] - ba[m][f"{f}{r}"] for f in ("R", "P", "S"))
            dists[f"DR_{m}_{r}"], dists[f"DP_{m}_{r}"], dists[f"DS_{m}_{r}"] = (
                d_r,
                d_p,
                d_s,
            )
            dists[f"I_{m}_{r}"] = d_r - d_p - d_s
    uni, v2 = ENCODERS[1], ENCODERS[0]
    for r in RATIOS_NEW:
        for x in ("DR", "DP", "DS", "I"):
            dists[f"delta_{x}_{r}"] = dists[f"{x}_{uni}_{r}"] - dists[f"{x}_{v2}_{r}"]
    dists["delta_B"] = ba[uni]["B"] - ba[v2]["B"]
    for r in RATIOS_NEW:
        dists[f"delta_R{r}_arm"] = ba[uni][f"R{r}"] - ba[v2][f"R{r}"]
    return dists
