"""Pure feature manipulations: moving class centres, splitting centre error, noise, whitening."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from centre import SHIFT_TOL

__all__ = [
    "CentreErrorParts",
    "class_means",
    "move_centres",
    "discriminant_basis",
    "centre_error_parts",
    "noise_shift",
    "between_patient_eigenbasis",
    "whiten",
]


class CentreErrorParts(NamedTuple):
    """Shared, class-specific discriminant, and class-specific off-discriminant parts (C, d)."""

    shared: np.ndarray
    discriminant: np.ndarray
    off_discriminant: np.ndarray


def class_means(x: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """(C, d) mean feature vector per class label."""
    return np.stack([x[y == c].mean(axis=0) for c in range(n_classes)])


def move_centres(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shift every class so its mean equals ``target[c]``, keeping each patch's offset from its centre.

    Raises if the achieved means or the offsets miss by more than ``SHIFT_TOL``.
    """
    means = class_means(x, y, len(target))
    moved = x + (target - means)[y]
    mean_err = np.abs(class_means(moved, y, len(target)) - target).max()
    offset_err = np.abs((moved - target[y]) - (x - means[y])).max()
    if mean_err > SHIFT_TOL or offset_err > SHIFT_TOL:
        raise RuntimeError(
            f"Centre shift failed: mean error {mean_err:.2e}, offset error {offset_err:.2e}"
        )
    return moved


def discriminant_basis(centres: np.ndarray) -> np.ndarray:
    """(C-1, d) orthonormal basis of the span of the centred class centres."""
    centred = centres - centres.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[: len(centres) - 1]


def centre_error_parts(error: np.ndarray, basis: np.ndarray) -> CentreErrorParts:
    """Split a (C, d) centre error into its class mean and the class-specific parts in and off ``basis``."""
    shared = np.repeat(error.mean(axis=0, keepdims=True), len(error), axis=0)
    specific = error - shared
    discriminant = (specific @ basis.T) @ basis
    return CentreErrorParts(shared, discriminant, specific - discriminant)


def noise_shift(deviations: np.ndarray, g: int, rng: np.random.Generator) -> np.ndarray:
    """(d,) class-centre error with covariance Sigma / g, Sigma from (n, d) patient deviations."""
    n = len(deviations)
    return deviations.T @ rng.standard_normal(n) / np.sqrt((n - 1) * g)


def between_patient_eigenbasis(
    deviations: np.ndarray, dof: int, rel_tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """(k, d) eigenvectors and (k,) eigenvalues of ``deviations.T @ deviations / dof`` above ``rel_tol * max``."""
    _, s, vt = np.linalg.svd(deviations, full_matrices=False)
    eigvals = s**2 / dof
    keep = eigvals > rel_tol * eigvals.max()
    return vt[keep], eigvals[keep]


def whiten(
    x: np.ndarray, basis: np.ndarray, eigvals: np.ndarray, kappa: float
) -> np.ndarray:
    """``x (I + kappa B)^(-1/2)`` for ``B = basis.T diag(eigvals) basis``, without forming d x d matrices."""
    scale = 1.0 / np.sqrt(1.0 + kappa * eigvals) - 1.0
    return x + ((x @ basis.T) * scale) @ basis
