from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from imbalance_benchmark.analysis.ordinal import ordinal_metrics

__all__ = [
    "assign_tiers",
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
    "classwise_nll",
    "classwise_brier",
    "tier_metrics",
    "classification_payload",
]

TIERS = ("head", "body", "tail")


def assign_tiers(
    class_names: list[str], allocated_counts: dict[str, int]
) -> dict[str, str]:
    """Rank classes by allocated training support into head/body/tail (report Eq. ceil(K/3)).

    Classes are ranked by allocated support, ties broken by the locked class
    order (``class_names`` itself, already alphabetically fixed). The first
    ceil(K/3) ranks are head, the last ceil(K/3) are tail, the rest are body;
    ``ceil(2/3) == 1`` makes the binary case (one head, one tail, no body)
    fall out of the same formula without special-casing.
    """
    k = len(class_names)
    n_edge = math.ceil(k / 3)
    order = sorted(
        range(k),
        key=lambda i: (-allocated_counts.get(class_names[i], 0), i),
    )
    tiers = {}
    for rank, idx in enumerate(order):
        if rank < n_edge:
            tiers[class_names[idx]] = "head"
        elif rank >= k - n_edge:
            tiers[class_names[idx]] = "tail"
        else:
            tiers[class_names[idx]] = "body"
    return tiers


def negative_log_likelihood(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Natural-prevalence NLL: mean negative log-probability of the true class."""
    if len(labels) == 0:
        return 0.0
    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def brier_score(labels: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    """Natural-prevalence multiclass Brier score."""
    if len(labels) == 0:
        return 0.0
    one_hot = np.eye(n_classes, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """Fixed-binning ECE (secondary probability-quality summary per the report)."""
    if len(labels) == 0:
        return 0.0
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(bins[:-1], bins[1:], strict=False):
        mask = (confidence > low) & (confidence <= high)
        if bool(mask.any()):
            accuracy = np.mean(predictions[mask] == labels[mask])
            error += float(mask.mean() * abs(accuracy - confidence[mask].mean()))
    return error


def classwise_nll(
    labels: np.ndarray, probabilities: np.ndarray, n_classes: int
) -> np.ndarray:
    """Per-class NLL restricted to samples of that true class; NaN where unsupported."""
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            out[c] = negative_log_likelihood(labels[mask], probabilities[mask])
    return out


def classwise_brier(
    labels: np.ndarray, probabilities: np.ndarray, n_classes: int
) -> np.ndarray:
    """Per-class Brier score restricted to samples of that true class; NaN where unsupported."""
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            out[c] = brier_score(labels[mask], probabilities[mask], n_classes)
    return out


def tier_metrics(
    class_names: list[str],
    tiers: dict[str, str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
    cw_nll: np.ndarray,
    cw_brier: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Average classwise precision/recall/f1/NLL/Brier within each locked tier."""
    out: dict[str, dict[str, Any]] = {}
    for tier in TIERS:
        idx = [i for i, name in enumerate(class_names) if tiers.get(name) == tier]
        if not idx:
            continue
        out[tier] = {
            "precision": float(np.mean(precision[idx])),
            "recall": float(np.mean(recall[idx])),
            "f1": float(np.mean(f1[idx])),
            "support": int(np.sum(support[idx])),
            "nll": float(np.nanmean(cw_nll[idx])),
            "brier": float(np.nanmean(cw_brier[idx])),
        }
    return out


def _per_class_stats(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, n_classes: int
) -> dict[str, np.ndarray]:
    """Precision/recall/F1/support plus classwise NLL/Brier, one entry per class."""
    label_range = list(range(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_range, zero_division=cast(str, 0)
    )
    return {
        "precision": cast(np.ndarray, precision),
        "recall": cast(np.ndarray, recall),
        "f1": cast(np.ndarray, f1),
        "support": cast(np.ndarray, support),
        "nll": classwise_nll(y_true, probs, n_classes),
        "brier": classwise_brier(y_true, probs, n_classes),
    }


def _scalar_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    stats: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Aggregate scalar discrimination/calibration summaries for one evaluated split."""
    present = cast(np.ndarray, stats["support"] > 0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(stats["precision"][present])),
        "macro_recall": float(np.mean(stats["recall"][present])),
        "macro_f1": float(np.mean(stats["f1"][present])),
        "negative_log_likelihood": negative_log_likelihood(y_true, probs),
        "macro_nll": float(np.nanmean(stats["nll"][present])),
        "brier_score": brier_score(y_true, probs, len(stats["support"])),
        "expected_calibration_error": expected_calibration_error(y_true, probs),
        **ordinal_metrics(y_true, y_pred),
    }


def _array_fields(
    y_true: np.ndarray, y_pred: np.ndarray, stats: dict[str, np.ndarray], n_classes: int
) -> dict[str, Any]:
    """Per-class arrays and the confusion matrix, JSON-serializable for the run record."""
    return {
        "precision_per_class": stats["precision"].tolist(),
        "recall_per_class": stats["recall"].tolist(),
        "f1_per_class": stats["f1"].tolist(),
        "support_per_class": stats["support"].tolist(),
        "nll_per_class": stats["nll"].tolist(),
        "brier_per_class": stats["brier"].tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))
        ).tolist(),
    }


def _coerce_arrays(
    labels: list[int] | np.ndarray,
    preds: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coerce raw label/prediction/probability inputs to typed numpy arrays."""
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(preds, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )


def _tier_payload(
    class_names: list[str], tiers: dict[str, str], stats: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Wrap ``tier_metrics`` with the stats dict's positional arguments it expects."""
    return tier_metrics(
        class_names,
        tiers,
        stats["precision"],
        stats["recall"],
        stats["f1"],
        stats["support"],
        stats["nll"],
        stats["brier"],
    )


def classification_payload(
    labels: list[int] | np.ndarray,
    preds: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray,
    class_names: list[str],
    tiers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Full discrimination + calibration payload for one evaluated split.

    Replaces the confirm-time stub that hard-wired macro precision/recall to
    balanced accuracy. Includes macro NLL (unweighted mean of class-conditional
    NLL) and, when ``tiers`` is given, head/body/tail-averaged Brier/NLL per
    report §"Endpoints and aggregation units" / §"Probability quality".
    """
    n_classes = len(class_names)
    y_true, y_pred, probs = _coerce_arrays(labels, preds, probabilities)
    stats = _per_class_stats(y_true, y_pred, probs, n_classes)
    payload = _scalar_metrics(y_true, y_pred, probs, stats)
    payload.update(_array_fields(y_true, y_pred, stats, n_classes))
    if tiers is not None:
        payload["tier_metrics"] = _tier_payload(class_names, tiers, stats)
    return payload
