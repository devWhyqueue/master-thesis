"""Class-recall model (report Eq. "model"): a shared estimator for analysis and simulation.

Every fit contains all classes, and the per-replicate weight is constant
within a fit (it resamples whole draws, not individual class rows). The
two-way fixed-effects (split-class, fit) demeaning is then exact in closed
form within each split: ``x - x_bar_f. - x_bar^w_.c + x_bar^w_..``, so no
dummy-variable regression is needed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate", "parts", "reading", "draw_weights"]


def _demean_split(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Two-way (fit, class) within-split demeaning of predictors and outcome.

    ``x`` is (F, C, 3), ``y`` is (F, C, R), ``w`` is (F, R). The fit mean is
    the plain class-average (``w`` does not vary within a fit); the class and
    grand means are weighted by ``w`` and computed within this split only,
    since the intercept is per (split, class).
    """
    f_bar_x = x.mean(axis=1)  # (F, 3)
    f_bar_y = y.mean(axis=1)  # (F, R)

    w_sum = w.sum(axis=0)  # (R,)
    c_bar_x = np.einsum("fr,fcj->crj", w, x) / w_sum[None, :, None]  # (C, R, 3)
    c_bar_y = np.einsum("fr,fcr->cr", w, y) / w_sum[None, :]  # (C, R)

    g_bar_x = np.einsum("fr,fj->rj", w, f_bar_x) / w_sum[:, None]  # (R, 3)
    g_bar_y = np.einsum("fr,fr->r", w, f_bar_y) / w_sum  # (R,)

    x_tilde = (
        x[:, :, None, :]
        - f_bar_x[:, None, None, :]
        - c_bar_x[None, :, :, :]
        + g_bar_x[None, None, :, :]
    )  # (F, C, R, 3)
    y_tilde = (
        y[:, :, :] - f_bar_y[:, None, :] - c_bar_y[None, :, :] + g_bar_y[None, None, :]
    )  # (F, C, R)
    return x_tilde, y_tilde


def estimate(
    y: np.ndarray, x: np.ndarray, fit_split: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Weighted two-way (split-class, fit) FE regression of Eq. "model", per replicate.

    ``y`` is (F, C, R) percent recall, ``x`` is (F, C, 3) the ``[r, omega,
    z]`` predictors, ``fit_split`` is (F,) the split of each fit, and ``w``
    is (F, R) frequency weights. Returns ``beta`` (R, 3): columns are
    delta, lambda, gamma.
    """
    _, _, n_replicates = y.shape
    a_total = np.zeros((n_replicates, 3, 3), dtype=np.float64)
    b_total = np.zeros((n_replicates, 3), dtype=np.float64)
    for s in np.unique(fit_split):
        mask = fit_split == s
        x_tilde, y_tilde = _demean_split(x[mask], y[mask], w[mask])
        ws = w[mask]
        a_total += np.einsum("fr,fcri,fcrj->rij", ws, x_tilde, x_tilde)
        b_total += np.einsum("fr,fcri,fcr->ri", ws, x_tilde, y_tilde)
    return np.linalg.solve(a_total, b_total[..., None])[..., 0]


def parts(
    beta: np.ndarray,
    r_bar_ran5: float,
    r_bar_ran10: float,
    omega_bar_ran5: float,
    omega_bar_ran10: float,
) -> dict[str, np.ndarray]:
    """The coverage, similarity, and patient-count parts of the gap (report Eq. "parts").

    ``beta`` is (R, 3); the random-cell coverage/similarity means are the
    fixed point estimates, not resampled per replicate.
    """
    delta, lam, gamma = beta[:, 0], beta[:, 1], beta[:, 2]
    return {
        "C": delta * (r_bar_ran10 - r_bar_ran5),
        "S": lam * (omega_bar_ran10 - omega_bar_ran5),
        "P": gamma,
    }


def reading(ci: tuple[float, float], threshold: float = 1.0) -> str:
    """One part's interpretation label (report Table "readings")."""
    lower, upper = ci
    if lower > threshold:
        return "contributes"
    if upper < -threshold:
        return "counteracts"
    if upper < threshold:
        return "at_most_small"
    return "unresolved"


def draw_weights(
    fit_split: np.ndarray, n_draws: int, n_replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Frequency weights (F, R) resampling draws within each split; column 0 is ones."""
    n_fits = len(fit_split)
    w = np.ones((n_fits, n_replicates), dtype=np.float64)
    for s in np.unique(fit_split):
        idx = np.flatnonzero(fit_split == s)
        if len(idx) != n_draws:
            raise ValueError(f"Split {s} has {len(idx)} fits, expected {n_draws}")
        counts = rng.multinomial(
            n_draws, np.full(n_draws, 1.0 / n_draws), size=n_replicates - 1
        )
        w[idx, 1:] = counts.T
    return w
