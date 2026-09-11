"""Support surface least-squares regression fits over the 3x3 grid."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "fit_ols",
    "fit_candidate_models",
]


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit OLS y = X @ theta with intercept column added automatically."""
    n_samples = len(y)
    x_matrix = np.column_stack([np.ones(n_samples), x])
    theta, _, _, _ = np.linalg.lstsq(x_matrix, y, rcond=None)

    preds = x_matrix @ theta
    sse = float(np.sum((y - preds) ** 2))
    df_resid = max(1, n_samples - x_matrix.shape[1])
    res_std = float(np.sqrt(sse / df_resid))

    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else 1.0
    return theta, res_std, r2


def _fit_single_models(
    log_n: np.ndarray, log_g: np.ndarray, log_neff: np.ndarray, accs: np.ndarray
) -> dict[str, dict[str, float]]:
    """Fit the three competing single-predictor surface models."""
    out = {}
    for name, pred in [("log_n", log_n), ("log_g", log_g), ("log_neff", log_neff)]:
        theta, res_std, r2 = fit_ols(pred[:, np.newaxis], accs)
        out[name] = {
            "alpha": float(theta[0]),
            "beta": float(theta[1]),
            "res_std": res_std,
            "r2": r2,
        }
    return out


def _fit_augmented_models(
    log_n: np.ndarray, log_g: np.ndarray, log_neff: np.ndarray, accs: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit augmented nominal and augmented effective models."""
    t_nom, s_nom, r2_nom = fit_ols(np.column_stack([log_n, log_g]), accs)
    aug_nom = {
        "alpha": float(t_nom[0]),
        "beta": float(t_nom[1]),
        "gamma_n": float(t_nom[2]),
        "res_std": s_nom,
        "r2": r2_nom,
    }
    t_eff, s_eff, r2_eff = fit_ols(np.column_stack([log_neff, log_g]), accs)
    aug_eff = {
        "alpha": float(t_eff[0]),
        "beta": float(t_eff[1]),
        "gamma_e": float(t_eff[2]),
        "res_std": s_eff,
        "r2": r2_eff,
    }
    return aug_nom, aug_eff


def fit_candidate_models(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    accuracies: np.ndarray,
    effective_supports: np.ndarray,
) -> dict[str, Any]:
    """Fit candidate surface predictors (Eq. 134) and augmented models."""
    g_vals = np.array([c[0] for c in cells], dtype=np.float64)
    m_vals = np.array([c[1] for c in cells], dtype=np.float64)

    log_n = np.log(g_vals * m_vals)
    log_g = np.log(g_vals)
    log_neff = np.log(np.maximum(1e-8, effective_supports))

    single = _fit_single_models(log_n, log_g, log_neff, accuracies)
    ranking = sorted(single.keys(), key=lambda k: single[k]["res_std"])
    aug_nom, aug_eff = _fit_augmented_models(log_n, log_g, log_neff, accuracies)

    return {
        "ranking": ranking,
        "single_models": single,
        "augmented_nominal": aug_nom,
        "augmented_effective": aug_eff,
    }
