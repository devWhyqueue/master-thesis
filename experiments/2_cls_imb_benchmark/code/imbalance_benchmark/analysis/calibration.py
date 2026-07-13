from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from imbalance_benchmark.analysis.metrics import negative_log_likelihood

__all__ = [
    "TemperatureFit",
    "probabilities_to_logits",
    "softmax",
    "apply_temperature",
    "fit_temperature",
    "reliability_curve",
    "estimate_prior",
    "apply_target_prior_correction",
    "balanced_decision_logits",
]

# Methods for which the report defines a target-prior correction (Eq.
# posthoc-target-prior / train-time-target-prior); every other method's raw
# score already is its balanced-decision score with no defined prior variant.
_TARGET_PRIOR_METHODS = frozenset({"post_hoc_logit_adjustment", "logit_adjustment"})


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
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, mean_confidence, accuracy = [], [], []
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

    Only ``post_hoc_logit_adjustment`` and ``logit_adjustment`` (train-time)
    have a defined target-prior variant in the report; every other method's
    target-prior-corrected output collapses to its raw score.
    """
    if method not in _TARGET_PRIOR_METHODS:
        return logits
    log_train = np.log(np.clip(pi_train, 1e-12, 1.0))
    log_target = np.log(np.clip(pi_target, 1e-12, 1.0))
    if method == "post_hoc_logit_adjustment":
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
