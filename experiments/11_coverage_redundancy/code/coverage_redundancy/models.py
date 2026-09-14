"""Split-intercept OLS design, per-replicate fitting, and cohort descriptives."""

from __future__ import annotations

from typing import Any

import numpy as np
from breadth.analyze.secondary import pack_estimate
from breadth.surface import fit_ols

from breadth import GRID_CELLS

from coverage_redundancy import COMPOSITION_ALLOCATIONS, LN2

__all__ = ["design", "fit_replicates", "model_result", "group_mean", "descriptives"]


def design(split_idx: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    """Two split-dummy columns (splits 1, 2; split 0 folds into the intercept)."""
    d1 = (split_idx == 1).astype(np.float64)
    d2 = (split_idx == 2).astype(np.float64)
    return np.column_stack([d1, d2, *columns])


def fit_replicates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one OLS design across every bootstrap replicate column of y (rows, R)."""
    n_reps = y.shape[1]
    thetas = np.empty((n_reps, x.shape[1] + 1))
    res_std = np.empty(n_reps)
    for i in range(n_reps):
        theta, res, _ = fit_ols(x, y[:, i])
        thetas[i] = theta
        res_std[i] = res
    return thetas, res_std


def model_result(
    theta: np.ndarray, res_std: np.ndarray, beta_idx: int, gamma_idx: int | None
) -> dict[str, Any]:
    """Pack one model's beta, optional gamma/b, and residual standard deviation."""
    out: dict[str, Any] = {
        "beta": pack_estimate(theta[:, beta_idx]),
        "res_std": pack_estimate(res_std),
    }
    if gamma_idx is not None:
        gamma = theta[:, gamma_idx]
        out["gamma"] = pack_estimate(gamma)
        out["b"] = pack_estimate(gamma * LN2)
    return out


def group_mean(values: np.ndarray, allocation: np.ndarray, name: str) -> np.ndarray:
    """Mean over one allocation's rows, per replicate column."""
    return values[allocation == name].mean(axis=0)


def _cell_descriptive(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean omega, Neff, Neff^omega, and r of one grid cell's random cohorts."""
    return {
        "omega_mean": float(np.mean([row["omega_mean"] for row in rows])),
        "neff": float(np.mean([np.exp(row["log_neff"]) for row in rows])),
        "neff_omega": float(np.mean([np.exp(row["log_neff_omega"]) for row in rows])),
        "r": float(np.mean([row["r"] for row in rows])),
    }


def _arm_descriptive(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean omega, Neff^omega, and r of one exp-10 arm's composition cohorts."""
    return {
        "omega_mean": float(np.mean([row["omega_mean"] for row in rows])),
        "neff_omega": float(np.mean([np.exp(row["log_neff_omega"]) for row in rows])),
        "r": float(np.mean([row["r"] for row in rows])),
    }


def descriptives(quantities: dict[str, Any]) -> dict[str, Any]:
    """Mean omega, Neff, Neff^omega, and r per grid cell and per exp-10 arm."""
    cohorts = quantities["cohorts"]
    cells = {
        f"G{g}_m{m}": _cell_descriptive(
            [
                c
                for c in cohorts
                if c["source"] == "random" and c["g"] == g and c["m"] == m
            ]
        )
        for g, m in GRID_CELLS
    }
    arms = {
        name: _arm_descriptive(
            [
                c
                for c in cohorts
                if c["source"] == "composition" and c["allocation"] == name
            ]
        )
        for name in COMPOSITION_ALLOCATIONS
    }
    return {"cells": cells, "composition_arms": arms}
