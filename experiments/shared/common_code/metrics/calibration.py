"""Calibration metrics."""

from __future__ import annotations

import numpy as np


def negative_log_likelihood(
    y_true: list[int], probabilities: list[list[float]], n_classes: int
) -> float:
    if not probabilities:
        return 0.0
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def brier_score(
    y_true: list[int], probabilities: list[list[float]], n_classes: int
) -> float:
    if not probabilities:
        return 0.0
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
    one_hot = np.eye(n_classes, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: list[int], probabilities: list[list[float]]
) -> float:
    if not probabilities:
        return 0.0
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
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


def calibration_metrics(
    y_true: list[int], probabilities: list[list[float]], n_classes: int
) -> dict[str, float]:
    if not probabilities:
        return {}
    return {
        "negative_log_likelihood": negative_log_likelihood(
            y_true, probabilities, n_classes
        ),
        "brier_score": brier_score(y_true, probabilities, n_classes),
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities
        ),
    }
