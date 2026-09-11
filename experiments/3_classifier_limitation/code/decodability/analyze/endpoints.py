"""Compute primary and secondary endpoints for decodability evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    clustered_endpoints,
)
from sklearn.metrics import balanced_accuracy_score

__all__ = [
    "compute_cell_endpoints",
    "per_class_patient_macro_recall",
]


def per_class_patient_macro_recall(
    labels: np.ndarray,
    predictions: np.ndarray,
    case_ids: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute per-class patient-macro recall vector.

    Note: clustered_endpoints computes this internally to produce the mean
    patient_macro_balanced_accuracy, but does not return the vector. We
    compute it here for the secondary endpoint.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    case_ids = np.asarray(case_ids)

    recalls = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        mask_c = labels == c
        if not np.any(mask_c):
            recalls[c] = 0.0
            continue
        c_cases = case_ids[mask_c]
        c_preds = predictions[mask_c]
        unique_cases = pd.unique(c_cases)
        patient_accuracies = [
            np.mean(c_preds[c_cases == patient] == c) for patient in unique_cases
        ]
        recalls[c] = float(np.mean(patient_accuracies))

    return recalls


def compute_cell_endpoints(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    identity: pd.DataFrame,
    n_classes: int,
) -> dict[str, Any]:
    """Compute primary, patch-micro, and per-class recall endpoints."""
    eps = 1e-12
    safe_probs = np.clip(probabilities, eps, 1.0)
    safe_probs = safe_probs / np.sum(safe_probs, axis=1, keepdims=True)

    clustered = clustered_endpoints(
        labels=labels,
        predictions=predictions,
        probabilities=safe_probs,
        identity=identity,
    )
    primary_ba = float(clustered["patient_macro_balanced_accuracy"])
    micro_ba = float(balanced_accuracy_score(labels, predictions))
    case_ids = identity["case_id"].astype(str).to_numpy()
    class_recalls = per_class_patient_macro_recall(
        labels, predictions, case_ids, n_classes
    )

    return {
        "patient_macro_balanced_accuracy": primary_ba,
        "patch_micro_balanced_accuracy": micro_ba,
        "per_class_recall": class_recalls.tolist(),
    }
