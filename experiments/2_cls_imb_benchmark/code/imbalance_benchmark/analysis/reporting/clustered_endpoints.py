from __future__ import annotations

import numpy as np
import pandas as pd

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


def _patient_bootstrap_ece(
    labels: np.ndarray, probabilities: np.ndarray, case_ids: np.ndarray, seed: int
) -> tuple[float, float]:
    """Return a patient-block percentile interval for fixed-bin ECE."""
    patients = np.unique(case_ids)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(1_000):
        draws = rng.choice(patients, len(patients), replace=True)
        rows = np.concatenate([np.flatnonzero(case_ids == case) for case in draws])
        samples.append(expected_calibration_error(labels[rows], probabilities[rows]))
    interval = np.percentile(samples, [2.5, 97.5])
    return float(interval[0]), float(interval[1])


def clustered_endpoints(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    identity: pd.DataFrame,
    seed: int,
) -> dict[str, float | list[float]]:
    """Compute patch-micro, slide/patient-macro, and clustered-ECE endpoints."""
    case_ids = identity["case_id"].astype(str).to_numpy()
    slide_ids = identity["slide_id"].astype(str).to_numpy()
    ece = expected_calibration_error(labels, probabilities)
    nll = _sample_nll(labels, probabilities)
    brier = _sample_brier(labels, probabilities)
    return {
        "patch_micro_accuracy": float(np.mean(predictions == labels)),
        "slide_macro_accuracy": _macro_accuracy(predictions, labels, slide_ids),
        "patient_macro_accuracy": _macro_accuracy(predictions, labels, case_ids),
        # Probability-quality contributions aggregated within each slide/patient
        # first, so heavily tiled slides do not dominate the summary.
        "slide_macro_nll": _macro_mean(nll, slide_ids),
        "patient_macro_nll": _macro_mean(nll, case_ids),
        "slide_macro_brier": _macro_mean(brier, slide_ids),
        "patient_macro_brier": _macro_mean(brier, case_ids),
        "expected_calibration_error": ece,
        "expected_calibration_error_ci": list(
            _patient_bootstrap_ece(labels, probabilities, case_ids, seed)
        ),
    }
