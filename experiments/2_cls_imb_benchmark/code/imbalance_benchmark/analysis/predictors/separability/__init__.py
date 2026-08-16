from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from imbalance_benchmark.analysis.predictors.separability.backend import (
    KNNConfig,
    ProbeData,
    knn_and_nn_probe as _backend_knn_and_nn_probe,
)

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


_CHUNK_SIZE = 4096
_REFERENCE_CHUNK_SIZE = 131072
logger = logging.getLogger(__name__)


def _knn_and_nn_probe(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_classes: int,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared streamed distance pass: k-NN vote predictions and 1-NN correctness.

    Both consumers share distances. Reference blocks bound the temporary matrix.
    """
    return _backend_knn_and_nn_probe(
        ProbeData(ref_x, ref_y, val_x, val_y, n_classes),
        KNNConfig(k, _CHUNK_SIZE, _REFERENCE_CHUNK_SIZE),
    )


def _macro_recall(preds: np.ndarray, val_y: np.ndarray, n_classes: int) -> float:
    """Class-balanced recall of ``preds`` against ``val_y`` over ``n_classes``."""
    recalls = [
        float((preds[mask] == c).mean())
        for c in range(n_classes)
        if (mask := val_y == c).any()
    ]
    return float(np.mean(recalls)) if recalls else 0.0


def balanced_knn_macro_recall(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_classes: int,
    k: int = 5,
) -> float:
    """Balanced (macro-recall) k-nearest-neighbour classification of the validation set."""
    preds, _ = _knn_and_nn_probe(ref_x, ref_y, val_x, val_y, n_classes, k)
    return _macro_recall(preds, val_y, n_classes)


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


def _per_class_error(
    nn_correct: np.ndarray, val_y: np.ndarray, n_classes: int
) -> np.ndarray:
    """Per-class 1-NN mismatch rate from a correctness mask; NaN where unsupported."""
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = val_y == c
        if mask.any():
            out[c] = 1.0 - float(nn_correct[mask].mean())
    return out


def per_class_nn_error(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Per-class 1-nearest-neighbour label-mismatch error rate; NaN where unsupported."""
    _, nn_correct = _knn_and_nn_probe(ref_x, ref_y, val_x, val_y, n_classes)
    return _per_class_error(nn_correct, val_y, n_classes)


def probe_metrics(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """The three fixed probes (kNN, linear, per-class NN error), scored on the validation set."""
    preds, nn_correct = _knn_and_nn_probe(ref_x, ref_y, val_x, val_y, n_classes)
    return {
        "knn_macro_recall": _macro_recall(preds, val_y, n_classes),
        "linear_probe_macro_recall": linear_probe_macro_recall(
            ref_x, ref_y, val_x, val_y
        ),
        "per_class_nn_error": _per_class_error(nn_correct, val_y, n_classes).tolist(),
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


def _margin_scores(
    probe: LogisticRegression, x: np.ndarray, y: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    """Return each held-out row's true-class logit minus its best competitor."""
    logits = probe.decision_function(x[test_idx])
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    true_logit = logits[np.arange(len(test_idx)), y[test_idx]]
    logits[np.arange(len(test_idx)), y[test_idx]] = -np.inf
    return true_logit - logits.max(axis=1)


def class_margin_cross_fit(
    x: np.ndarray,
    y: np.ndarray,
    case_ids: np.ndarray,
    n_classes: int,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Grouped cross-fitted true-class-minus-best-competitor logit margin."""
    del seed  # GroupKFold is deterministic; retain the public cross-fit signature.
    margins = np.full(len(y), np.nan)
    n_folds = max(2, min(n_folds, len(np.unique(case_ids))))
    splitter = GroupKFold(n_splits=n_folds)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, case_ids), 1):
        logger.info("rq3: margin fold %d/%d", fold, n_folds)
        if len(np.unique(y[train_idx])) < 2:
            continue
        started = time.monotonic()
        probe = LogisticRegression(class_weight="balanced", max_iter=1000)
        probe.fit(x[train_idx], y[train_idx])
        margins[test_idx] = _margin_scores(probe, x, y, test_idx)
        logger.info(
            "rq3: margin fold %d/%d complete in %.1fs (%d lbfgs iterations)",
            fold,
            n_folds,
            time.monotonic() - started,
            int(np.max(probe.n_iter_)),
        )
    return margins


def intraclass_correlation(margin: np.ndarray, cluster_ids: np.ndarray) -> float:
    """ANOVA ICC(1), including Fisher's unbalanced-group correction."""
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
