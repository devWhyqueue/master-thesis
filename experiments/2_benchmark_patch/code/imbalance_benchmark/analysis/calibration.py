from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from imbalance_benchmark.analysis.metrics import (
    brier_score,
    confidence_bin_index,
    expected_calibration_error,
    negative_log_likelihood,
)

__all__ = [
    "TemperatureFit",
    "probabilities_to_logits",
    "softmax",
    "apply_temperature",
    "temperature_scaled_probabilities",
    "fit_temperature",
    "reliability_curve",
    "seed_averaged_reliability_curve",
    "estimate_prior",
    "apply_target_prior_correction",
    "balanced_decision_logits",
    "temperature_scaled_payload",
]

# Methods for which the report defines a target-prior correction (Eq.
# posthoc-target-prior / train-time-target-prior); every other method's raw
# score already is its balanced-decision score with no defined prior variant.
_TARGET_PRIOR_METHODS = frozenset(
    {"ce", "post_hoc_logit_adjustment", "logit_adjustment"}
)


@dataclass(frozen=True)
class TemperatureFit:
    """A single fitted post-hoc temperature scalar."""

    temperature: float


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


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> TemperatureFit:
    """Fit one positive temperature scalar on validation logits by minimizing NLL.

    Per report §"Training, selection, and replication": fit on natural-
    validation NLL after model/checkpoint selection (and after the target-
    prior correction, when one applies), applied unchanged to test logits.
    """

    def _objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probs = apply_temperature(logits, temperature)
        return negative_log_likelihood(labels, probs)

    optimum = minimize_scalar(_objective, method="brent")
    return TemperatureFit(temperature=float(np.exp(optimum.x)))


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centers, mean confidence, and accuracy per confidence bin."""
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    bin_of_row = confidence_bin_index(confidence, n_bins)
    centers, mean_confidence, accuracy = [], [], []
    for b in range(n_bins):
        mask = bin_of_row == b
        if not bool(mask.any()):
            continue
        centers.append((b + 0.5) / n_bins)
        mean_confidence.append(float(confidence[mask].mean()))
        accuracy.append(float(np.mean(predictions[mask] == labels[mask])))
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(mean_confidence, dtype=np.float64),
        np.asarray(accuracy, dtype=np.float64),
    )


def seed_averaged_reliability_curve(
    probability_stack: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average fixed-bin reliability summaries over initialization seeds."""
    confidence = np.full((len(probability_stack), n_bins), np.nan)
    accuracy = np.full_like(confidence, np.nan)
    for seed, probabilities in enumerate(probability_stack):
        centers, seed_confidence, seed_accuracy = reliability_curve(
            probabilities, labels, n_bins
        )
        indices = (centers * n_bins).astype(int)
        confidence[seed, indices] = seed_confidence
        accuracy[seed, indices] = seed_accuracy
    present = ~np.isnan(confidence).all(axis=0)
    centers = (np.flatnonzero(present) + 0.5) / n_bins
    return (
        centers,
        np.nanmean(confidence[:, present], axis=0),
        np.nanmean(accuracy[:, present], axis=0),
    )


def temperature_scaled_payload(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, object]:
    """Fit validation temperature and retain scalar test calibration outputs."""
    fit = fit_temperature(validation_logits, validation_labels)
    probabilities = apply_temperature(test_logits, fit.temperature)
    centers, confidence, accuracy = reliability_curve(probabilities, test_labels)
    metrics = _temperature_metrics(test_labels, probabilities)
    return {
        "temperature": fit.temperature,
        **metrics,
        "temperature_scaled_reliability": {
            "bin_centers": centers.tolist(),
            "mean_confidence": confidence.tolist(),
            "accuracy": accuracy.tolist(),
        },
    }


def temperature_scaled_probabilities(payload: dict[str, object]) -> np.ndarray:
    """Load legacy scaled probabilities or reconstruct them from stored logits."""
    if "temperature_scaled_probabilities" in payload:
        return np.asarray(payload["temperature_scaled_probabilities"])
    temperature = payload.get("temperature")
    logits = payload.get("logits")
    if isinstance(temperature, (int, float)) and logits is not None:
        return apply_temperature(np.asarray(logits), float(temperature))
    return np.asarray(payload["probabilities"])


def _temperature_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Return temperature-scaled calibration metrics and patient-block ECE uncertainty."""
    return {
        "temperature_scaled_nll": negative_log_likelihood(labels, probabilities),
        "temperature_scaled_brier": brier_score(
            labels, probabilities, probabilities.shape[1]
        ),
        "temperature_scaled_ece": expected_calibration_error(labels, probabilities),
    }


def estimate_prior(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Estimate a class-proportion prior from an integer label array."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.full(n_classes, 1.0 / n_classes)
    return counts / total


def apply_target_prior_correction(
    logits: np.ndarray,
    method: str,
    tau: float,
    pi_train: np.ndarray,
    pi_target: np.ndarray,
) -> np.ndarray:
    """Return target-prevalence logits per Eq. posthoc/train-time-target-prior.

    Ordinary CE, post-hoc adjustment, and train-time adjustment have defined
    target-prior variants. Every other method retains its raw score.
    """
    if method not in _TARGET_PRIOR_METHODS:
        return logits
    log_train = np.log(np.clip(pi_train, 1e-12, 1.0))
    log_target = np.log(np.clip(pi_target, 1e-12, 1.0))
    if method in {"ce", "post_hoc_logit_adjustment"}:
        return logits - log_train + log_target
    return logits + (tau - 1.0) * log_train + log_target


def balanced_decision_logits(
    logits: np.ndarray, method: str, tau: float, pi_train: np.ndarray
) -> np.ndarray:
    """Return the score used for balanced-accuracy decisions (Eq. posthoc-target-prior).

    Post-hoc logit adjustment shifts the CE baseline by ``tau*log(pi_train)``;
    every other method (including train-time logit adjustment, whose raw score
    is already its validation-selected discrimination score) uses its raw score.
    """
    if method != "post_hoc_logit_adjustment":
        return logits
    log_train = np.log(np.clip(pi_train, 1e-12, 1.0))
    return logits - tau * log_train
