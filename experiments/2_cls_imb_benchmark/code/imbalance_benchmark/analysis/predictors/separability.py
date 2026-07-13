from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

__all__ = [
    "balanced_knn_macro_recall",
    "linear_probe_macro_recall",
    "per_class_nn_error",
    "probe_metrics",
    "intrinsic_separability",
    "condition_learnability",
    "class_margin_cross_fit",
    "intraclass_correlation",
    "effective_support",
]


def _pairwise_nearest(ref_x: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    """Index of each query row's nearest reference row by squared Euclidean distance."""
    d2 = (
        (query_x**2).sum(axis=1, keepdims=True)
        - 2.0 * query_x @ ref_x.T
        + (ref_x**2).sum(axis=1)[None, :]
    )
    return d2.argmin(axis=1)


def balanced_knn_macro_recall(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_classes: int,
    k: int = 5,
) -> float:
    """Balanced (macro-recall) k-nearest-neighbour classification of the validation set."""
    d2 = (
        (val_x**2).sum(axis=1, keepdims=True)
        - 2.0 * val_x @ ref_x.T
        + (ref_x**2).sum(axis=1)[None, :]
    )
    k = min(k, ref_x.shape[0])
    neighbor_idx = np.argsort(d2, axis=1)[:, :k]
    neighbor_labels = ref_y[neighbor_idx]
    preds = np.array(
        [np.bincount(row, minlength=n_classes).argmax() for row in neighbor_labels]
    )
    recalls = []
    for c in range(n_classes):
        mask = val_y == c
        if mask.any():
            recalls.append(float((preds[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def linear_probe_macro_recall(
    ref_x: np.ndarray, ref_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray
) -> float:
    """Class-balanced linear probe (logistic regression), macro recall on the validation set."""
    if len(np.unique(ref_y)) < 2:
        return 0.0
    probe = LogisticRegression(class_weight="balanced", max_iter=1000)
    probe.fit(ref_x, ref_y)
    preds = probe.predict(val_x)
    recalls = []
    for c in np.unique(val_y):
        mask = val_y == c
        recalls.append(float((preds[mask] == c).mean()))
    return float(np.mean(recalls))


def per_class_nn_error(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Per-class 1-nearest-neighbour label-mismatch error rate; NaN where unsupported."""
    nearest = _pairwise_nearest(ref_x, val_x)
    correct = ref_y[nearest] == val_y
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = val_y == c
        if mask.any():
            out[c] = 1.0 - float(correct[mask].mean())
    return out


def probe_metrics(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """The three fixed probes (kNN, linear, per-class NN error), scored on the validation set."""
    return {
        "knn_macro_recall": balanced_knn_macro_recall(
            ref_x, ref_y, val_x, val_y, n_classes
        ),
        "linear_probe_macro_recall": linear_probe_macro_recall(
            ref_x, ref_y, val_x, val_y
        ),
        "per_class_nn_error": per_class_nn_error(
            ref_x, ref_y, val_x, val_y, n_classes
        ).tolist(),
    }


def intrinsic_separability(
    balanced_reference_x: np.ndarray,
    balanced_reference_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """Intrinsic separability: fixed balanced reference subset, shared by every condition."""
    return probe_metrics(
        balanced_reference_x, balanced_reference_y, val_x, val_y, n_classes
    )


def condition_learnability(
    condition_x: np.ndarray,
    condition_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """Condition-specific learnability: the actual controlled training manifest as reference."""
    return probe_metrics(condition_x, condition_y, val_x, val_y, n_classes)


def class_margin_cross_fit(
    x: np.ndarray,
    y: np.ndarray,
    case_ids: np.ndarray,
    n_classes: int,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Cross-fitted scalar class margin: the intrinsic probe's true-class logit minus its best competitor.

    Folds are grouped by patient so an observation is never scored by a probe
    fitted on its own patient's data, matching Eq. neff's cross-fitting
    requirement.
    """
    margins = np.full(len(y), np.nan)
    n_folds = max(2, min(n_folds, len(np.unique(case_ids))))
    splitter = GroupKFold(n_splits=n_folds)
    for train_idx, test_idx in splitter.split(x, y, groups=case_ids):
        if len(np.unique(y[train_idx])) < 2:
            continue
        probe = LogisticRegression(class_weight="balanced", max_iter=1000)
        probe.fit(x[train_idx], y[train_idx])
        logits = probe.decision_function(x[test_idx])
        if logits.ndim == 1:
            logits = np.stack([-logits, logits], axis=1)
        true_logit = logits[np.arange(len(test_idx)), y[test_idx]]
        masked = logits.copy()
        masked[np.arange(len(test_idx)), y[test_idx]] = -np.inf
        best_competitor = masked.max(axis=1)
        margins[test_idx] = true_logit - best_competitor
    return margins


def intraclass_correlation(margin: np.ndarray, cluster_ids: np.ndarray) -> float:
    """One-way random-effects ICC(1) of a scalar measurement across clusters (patients/slides).

    Standard ANOVA-based estimator with Fisher's unbalanced-group correction
    for the average cluster size ``m0``.
    """
    valid = ~np.isnan(margin)
    margin, cluster_ids = margin[valid], cluster_ids[valid]
    groups = {c: margin[cluster_ids == c] for c in np.unique(cluster_ids)}
    k = len(groups)
    if k < 2:
        return 0.0
    n_total = len(margin)
    grand_mean = margin.mean()
    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups.values())
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups.values())
    dfb, dfw = k - 1, n_total - k
    if dfw <= 0 or dfb <= 0:
        return 0.0
    msb, msw = ssb / dfb, ssw / dfw
    sizes = np.array([len(g) for g in groups.values()])
    m0 = (n_total - (sizes**2).sum() / n_total) / dfb
    if m0 <= 0 or msb + (m0 - 1) * msw == 0:
        return 0.0
    icc = (msb - msw) / (msb + (m0 - 1) * msw)
    return float(np.clip(icc, 0.0, 1.0))


def effective_support(n_c: int, mean_cluster_size: float, icc: float) -> float:
    """N_eff,c = N_c / (1 + (mbar_c - 1) * ICC_c) (Eq. neff)."""
    denom = 1.0 + (mean_cluster_size - 1.0) * icc
    return n_c / denom if denom > 0 else float(n_c)
