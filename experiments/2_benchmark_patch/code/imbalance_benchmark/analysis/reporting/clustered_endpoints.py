from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from imbalance_benchmark.analysis.metrics import expected_calibration_error

__all__ = ["clustered_endpoints"]


def _macro_accuracy(
    predictions: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> float:
    """Return equal-weight mean accuracy after aggregating within each cluster."""
    correct = predictions == labels
    return float(pd.Series(correct).groupby(groups, sort=False).mean().mean())


def _sample_nll(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Per-sample negative log-likelihood of the true class."""
    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return -np.log(clipped)


def _sample_brier(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Per-sample multiclass Brier contribution."""
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return np.sum((probabilities - one_hot) ** 2, axis=1)


def _macro_mean(contribution: np.ndarray, groups: np.ndarray) -> float:
    """Aggregate a per-sample contribution within each cluster, then average equally."""
    return float(pd.Series(contribution).groupby(groups, sort=False).mean().mean())


def _macro_classification(
    labels: np.ndarray, predictions: np.ndarray, groups: np.ndarray
) -> tuple[float, float]:
    """Equal-weight balanced accuracy and macro F1 after cluster aggregation."""
    recalls = []
    for label in np.unique(labels):
        class_rows = labels == label
        correct = predictions[class_rows] == labels[class_rows]
        recalls.append(
            float(
                pd.Series(correct).groupby(groups[class_rows], sort=False).mean().mean()
            )
        )
    scores = [
        f1_score(
            labels[groups == group],
            predictions[groups == group],
            average="macro",
            zero_division=0,  # type: ignore
        )
        for group in pd.unique(groups)
    ]
    return float(np.mean(recalls)), float(np.mean(scores))


def clustered_endpoints(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    identity: pd.DataFrame,
    seed: int | None = None,
    is_mil: bool = False,
) -> dict[str, float]:
    """Compute regime-applicable accuracy, cluster-macro, and ECE endpoints."""
    del seed
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    case_ids = identity["case_id"].astype(str).to_numpy()
    slide_ids = identity["slide_id"].astype(str).to_numpy()
    ece = expected_calibration_error(labels, probabilities)
    nll = _sample_nll(labels, probabilities)
    brier = _sample_brier(labels, probabilities)
    endpoints = _cluster_discrimination(
        labels, predictions, case_ids, slide_ids, is_mil
    )
    return {
        **endpoints,
        # Probability-quality contributions aggregated within each slide/patient
        # first, so heavily tiled slides do not dominate the summary.
        "slide_macro_nll": _macro_mean(nll, slide_ids),
        "patient_macro_nll": _macro_mean(nll, case_ids),
        "slide_macro_brier": _macro_mean(brier, slide_ids),
        "patient_macro_brier": _macro_mean(brier, case_ids),
        "expected_calibration_error": ece,
    }


def _cluster_discrimination(
    labels: np.ndarray,
    predictions: np.ndarray,
    case_ids: np.ndarray,
    slide_ids: np.ndarray,
    is_mil: bool,
) -> dict[str, float]:
    """Compute regime-applicable micro and cluster-macro discrimination."""
    slide_ba, slide_f1 = _macro_classification(labels, predictions, slide_ids)
    patient_ba, patient_f1 = _macro_classification(labels, predictions, case_ids)
    result = {
        "slide_macro_accuracy": _macro_accuracy(predictions, labels, slide_ids),
        "patient_macro_accuracy": _macro_accuracy(predictions, labels, case_ids),
        "slide_macro_balanced_accuracy": slide_ba,
        "patient_macro_balanced_accuracy": patient_ba,
        "slide_macro_f1": slide_f1,
        "patient_macro_f1": patient_f1,
    }
    if not is_mil:
        result["patch_micro_accuracy"] = float(np.mean(predictions == labels))
    return result
