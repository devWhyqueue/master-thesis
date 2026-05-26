"""Post-hoc probability calibration helpers for stored val/test result JSON."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from scripts.mil.metrics import _calibration_metrics

N_BINS = 10


@dataclass(frozen=True)
class CalibrationFit:
    """Parameters for one post-hoc calibrator."""

    name: str
    temperature: float | None = None
    weights: np.ndarray | None = None
    biases: np.ndarray | None = None


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    """Recover logits up to an additive constant from a softmax probability matrix."""
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return np.log(clipped)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Apply row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return temperature-scaled probabilities."""
    return softmax(logits / max(temperature, 1e-6))


def apply_vector_scaling(
    logits: np.ndarray, weights: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    """Return vector-scaled probabilities (Guo et al., extension of temperature scaling)."""
    scaled = logits * weights + biases
    return softmax(scaled)


def apply_dirichlet_scaling(
    logits: np.ndarray, matrix: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    """Return Dirichlet-calibrated probabilities (Kull et al.; netcal-style linear map)."""
    alpha = np.exp(np.clip(logits @ matrix.T + biases, -30.0, 30.0))
    return alpha / alpha.sum(axis=1, keepdims=True)


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Average NLL for one probability matrix."""
    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> CalibrationFit:
    """Fit a single temperature on validation logits by minimizing NLL."""

    def _objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probs = apply_temperature(logits, temperature)
        return negative_log_likelihood(probs, labels)

    optimum = minimize_scalar(_objective, method="brent")
    temperature = float(np.exp(optimum.x))
    return CalibrationFit("temperature", temperature=temperature)


def fit_vector_scaling(logits: np.ndarray, labels: np.ndarray) -> CalibrationFit:
    """Fit per-class weights and biases on validation logits."""
    n_classes = logits.shape[1]
    initial = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])

    def _objective(params: np.ndarray) -> float:
        weights = params[:n_classes]
        biases = params[n_classes:]
        probs = apply_vector_scaling(logits, weights, biases)
        return negative_log_likelihood(probs, labels)

    optimum = minimize(_objective, initial, method="L-BFGS-B")
    weights = optimum.x[:n_classes]
    biases = optimum.x[n_classes:]
    return CalibrationFit("vector", weights=weights, biases=biases)


def fit_dirichlet(logits: np.ndarray, labels: np.ndarray) -> CalibrationFit:
    """Fit a linear Dirichlet calibrator on validation logits."""
    n_classes = logits.shape[1]
    initial = np.concatenate([np.eye(n_classes).reshape(-1), np.zeros(n_classes)])

    def _objective(params: np.ndarray) -> float:
        matrix = params[: n_classes * n_classes].reshape(n_classes, n_classes)
        biases = params[n_classes * n_classes :]
        probs = apply_dirichlet_scaling(logits, matrix, biases)
        return negative_log_likelihood(probs, labels)

    optimum = minimize(_objective, initial, method="L-BFGS-B", maxiter=250)
    matrix = optimum.x[: n_classes * n_classes].reshape(n_classes, n_classes)
    biases = optimum.x[n_classes * n_classes :]
    return CalibrationFit("dirichlet", weights=matrix, biases=biases)


def calibrated_probabilities(logits: np.ndarray, fit: CalibrationFit) -> np.ndarray:
    """Apply a fitted calibrator to logits."""
    if fit.name == "temperature":
        if fit.temperature is None:
            raise ValueError("Temperature fit is missing temperature.")
        return apply_temperature(logits, fit.temperature)
    if fit.name == "vector":
        if fit.weights is None or fit.biases is None:
            raise ValueError("Vector fit is missing weights or biases.")
        return apply_vector_scaling(logits, fit.weights, fit.biases)
    if fit.name == "dirichlet":
        if fit.weights is None or fit.biases is None:
            raise ValueError("Dirichlet fit is missing matrix or biases.")
        return apply_dirichlet_scaling(logits, fit.weights, fit.biases)
    raise ValueError(f"Unsupported calibrator: {fit.name}")


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centers, mean confidence, and accuracy per confidence bin."""
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    mean_confidence = []
    accuracy = []
    for low, high in zip(bins[:-1], bins[1:], strict=False):
        mask = (confidence > low) & (confidence <= high)
        if not bool(mask.any()):
            continue
        centers.append(0.5 * (low + high))
        mean_confidence.append(float(confidence[mask].mean()))
        accuracy.append(float(np.mean(predictions[mask] == labels[mask])))
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(mean_confidence, dtype=np.float64),
        np.asarray(accuracy, dtype=np.float64),
    )


def metric_bundle(
    probabilities: np.ndarray, labels: np.ndarray, n_classes: int
) -> dict[str, float]:
    """Return NLL, Brier, and ECE for one probability matrix."""
    metrics = _calibration_metrics(labels.tolist(), probabilities.tolist(), n_classes)
    return {
        "negative_log_likelihood": float(metrics["negative_log_likelihood"]),
        "brier_score": float(metrics["brier_score"]),
        "expected_calibration_error": float(metrics["expected_calibration_error"]),
    }
