"""Patient-macro weighting, confusion, score margins, and prediction transitions.

Weights follow the primary endpoint (``imbalance_benchmark.analysis.inference.bootstrap``'s
``case_class_divisor``): patches are averaged within each (patient, true class) group, groups
within their class, and classes equally. A BRACS patient with two lesion classes therefore forms
two groups, never one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "patient_weights",
    "confusion",
    "margins",
    "transitions",
    "weighted_cdf",
    "weighted_median",
    "lost_by_margin",
]

_PROBABILITY_FLOOR = np.finfo(np.float64).tiny
_MASS_TOLERANCE = 1e-9


def patient_weights(labels: np.ndarray, patients: np.ndarray) -> np.ndarray:
    """Per-row weight with unit mass per class, split equally over its patient-class groups."""
    key = pd.DataFrame({"patient": np.asarray(patients).astype(str), "label": labels})
    rows_in_group = key.groupby(["patient", "label"])["label"].transform("size")
    groups_in_class = key.groupby("label")["patient"].transform("nunique")
    return 1.0 / (rows_in_group.to_numpy(float) * groups_in_class.to_numpy(float))


def confusion(
    labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray, n_classes: int
) -> np.ndarray:
    """(K, K) row-normalised confusion; row t, column p is the weighted share of t predicted p."""
    out = np.zeros((n_classes, n_classes))
    np.add.at(out, (labels, predictions), weights)
    present = np.bincount(labels, minlength=n_classes) > 0
    if not np.allclose(out.sum(axis=1)[present], 1.0, atol=_MASS_TOLERANCE):
        raise ValueError("weights must give every present class unit mass")
    return out


def margins(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """True-class log-probability minus the strongest rival's, and the exact-tie mask.

    A zero margin is a tie: argmax resolves it by index order, so it is reported rather than
    counted as either correct or incorrect. Probabilities are floored at the smallest positive
    double before the log, so an underflowed rival stays finite.
    """
    log_p = np.log(np.maximum(probabilities, _PROBABILITY_FLOOR))
    rows = np.arange(len(labels))
    true = log_p[rows, labels]
    rivals = log_p.copy()
    rivals[rows, labels] = -np.inf
    margin = true - rivals.max(axis=1)
    return margin, margin == 0.0


def transitions(
    reference: np.ndarray, changed: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Weighted correct predictions lost and gained from ``reference`` to ``changed``.

    Per class, lost minus gained equals the recall drop; their class means reproduce the
    balanced-accuracy damage exactly.
    """
    n_classes = int(labels.max()) + 1
    ref_ok, chg_ok = reference == labels, changed == labels
    lost = np.bincount(labels, weights * (ref_ok & ~chg_ok), minlength=n_classes)
    gained = np.bincount(labels, weights * (~ref_ok & chg_ok), minlength=n_classes)
    return {
        "lost": float(lost.mean()),
        "gained": float(gained.mean()),
        "lost_per_class": lost,
        "gained_per_class": gained,
    }


def weighted_cdf(
    values: np.ndarray, weights: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Weighted share of ``values`` at or below each grid point, normalised to total mass one."""
    order = np.argsort(values, kind="stable")
    mass = np.cumsum(weights[order]) / weights.sum()
    position = np.searchsorted(values[order], grid, side="right")
    return np.where(position > 0, mass[np.maximum(position - 1, 0)], 0.0)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Smallest value whose weighted CDF reaches one half."""
    order = np.argsort(values, kind="stable")
    mass = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(mass, 0.5)])


def lost_by_margin(
    margin: np.ndarray,
    reference_ok: np.ndarray,
    changed_ok: np.ndarray,
    weights: np.ndarray,
    edges: tuple[float, ...],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per reference-margin bin (lo, hi]: reference-correct mass and the part of it lost."""
    bins = np.searchsorted(np.asarray(edges), margin, side="left") - 1
    n_bins = len(edges) - 1
    keep = reference_ok & (bins >= 0) & (bins < n_bins)
    correct = np.bincount(bins[keep], weights[keep], minlength=n_bins)
    lost_rows = keep & ~changed_ok
    lost = np.bincount(bins[lost_rows], weights[lost_rows], minlength=n_bins)
    return correct / n_classes, lost / n_classes
