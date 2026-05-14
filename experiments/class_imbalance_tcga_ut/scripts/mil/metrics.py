from __future__ import annotations

import numpy as np


def extra_metrics(
    y_true: list[int],
    probabilities: list[list[float]],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
    n_classes: int,
) -> dict[str, object]:
    """Build support-tier and probability-quality metrics."""
    payload = _tier_metrics(precision, recall, f1, support)
    payload.update(_calibration_metrics(y_true, probabilities, n_classes))
    return payload


def _tier_metrics(
    precision: np.ndarray, recall: np.ndarray, f1: np.ndarray, support: np.ndarray
) -> dict[str, object]:
    order = np.argsort(support)
    tiers = {
        "tail": order[:8],
        "body": order[8:-8] if len(order) > 16 else order,
        "head": order[-8:],
    }
    return {
        "support_tier_metrics": {
            name: {
                "precision": float(np.mean(precision[index])),
                "recall": float(np.mean(recall[index])),
                "f1": float(np.mean(f1[index])),
                "support": int(np.sum(support[index])),
            }
            for name, index in tiers.items()
            if len(index) > 0
        }
    }


def _calibration_metrics(
    y_true: list[int], probabilities: list[list[float]], n_classes: int
) -> dict[str, object]:
    if not probabilities:
        return {}
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    one_hot = np.eye(n_classes, dtype=np.float64)[labels]
    return {
        "negative_log_likelihood": float(-np.mean(np.log(clipped))),
        "brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "expected_calibration_error": _expected_calibration_error(probs, labels),
    }


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray) -> float:
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, 11)
    error = 0.0
    for low, high in zip(bins[:-1], bins[1:], strict=False):
        mask = (confidence > low) & (confidence <= high)
        if bool(mask.any()):
            accuracy = np.mean(predictions[mask] == labels[mask])
            error += float(mask.mean() * abs(accuracy - confidence[mask].mean()))
    return error
