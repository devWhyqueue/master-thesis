"""The precision simulation's mechanics: one study, its half-widths, and draw-count selection.

A simulated study resamples ``draws`` whole checked draws (with replacement,
within split) from the census's 20 checked draws, reusing each resampled
draw's own achieved cell, coverage distance, and similarity for all 30
classes together. Class recall is the pooled class mean for that patient
count, plus a training-draw deviation independently resampled -- per class,
split, and patient count -- from the matching stored historical fits, plus
the scenario's imposed effects, plus test-patient bootstrap noise read from
that same chosen historical fit's own replicate columns (the replicate
*index* is shared across classes and cells; the underlying fit is not).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.analysis.inference.gates import confidence_interval

from decomposition import (
    DRAW_COUNTS,
    HALFWIDTH_TOL_PP,
    N_DRAWS_DEFAULT,
    N_SIM,
    N_SPLITS,
    N_STUDY_REPLICATES,
    SCENARIO_C,
    SCENARIO_P,
    SCENARIO_S,
)
from decomposition.model import draw_weights, estimate, parts

__all__ = ["CheckedSplit", "select_draws"]

_DISPERSIONS: tuple[float, ...] = (1.0, 1.5)


class CheckedSplit(NamedTuple):
    """One split's 20 checked draws as (draw, class)-indexed arrays."""

    r_val: np.ndarray  # (20, C)
    omega: np.ndarray  # (20, C)
    g: np.ndarray  # (20, C) int
    r_level: np.ndarray  # (20, C) int, -1 for the two random cells


def _one_study(
    rng: np.random.Generator,
    checked: list[CheckedSplit],
    pooled_mean: dict[int, np.ndarray],
    pool_stacks: dict[int, list[np.ndarray]],
    n_draws: int,
    n_classes: int,
    delta: float,
    lam: float,
    gamma: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, float], dict[int, float]]:
    """One simulated study's (x, y, fit_split) and its random-cell coverage/similarity means."""
    n_fits = N_SPLITS * n_draws
    x = np.empty((n_fits, n_classes, 3), dtype=np.float64)
    y = np.empty((n_fits, n_classes, N_STUDY_REPLICATES), dtype=np.float64)
    fit_split = np.repeat(np.arange(N_SPLITS), n_draws)
    random_r: dict[int, list[float]] = {5: [], 10: []}
    random_omega: dict[int, list[float]] = {5: [], 10: []}
    c_grid = np.broadcast_to(np.arange(n_classes), (n_draws, n_classes))

    for s in range(N_SPLITS):
        slots = rng.integers(0, N_DRAWS_DEFAULT, size=n_draws)
        r_val = checked[s].r_val[slots]  # (D, C)
        omega = checked[s].omega[slots]
        g = checked[s].g[slots]
        r_level = checked[s].r_level[slots]
        z = (g == 10).astype(np.float64)

        fit_recall: dict[int, np.ndarray] = {}
        for patient_count, pool in pool_stacks.items():
            chosen = rng.integers(0, pool[s].shape[0], size=(n_draws, n_classes))
            fit_recall[patient_count] = pool[s][chosen, c_grid]  # (D, C, R_template)

        is_g10 = g == 10
        recall = np.where(is_g10[..., None], fit_recall[10], fit_recall[5])
        mean_sel = np.where(is_g10, pooled_mean[10][None, :], pooled_mean[5][None, :])

        training_dev = kappa * (recall[..., 0] - mean_sel)
        imposed = delta * r_val + lam * omega + gamma * z
        base = mean_sel + training_dev + imposed  # (D, C)

        cols = recall[..., :N_STUDY_REPLICATES]  # (D, C, R)
        y_slot = base[..., None] + (cols - cols[..., :1])
        y_slot[..., 0] = base

        f0 = s * n_draws
        x[f0 : f0 + n_draws, :, 0] = r_val
        x[f0 : f0 + n_draws, :, 1] = omega
        x[f0 : f0 + n_draws, :, 2] = z
        y[f0 : f0 + n_draws] = y_slot

        for patient_count in (5, 10):
            mask = (g == patient_count) & (r_level == -1)
            random_r[patient_count].extend(r_val[mask].tolist())
            random_omega[patient_count].extend(omega[mask].tolist())

    r_bar = {k: float(np.mean(v)) for k, v in random_r.items()}
    omega_bar = {k: float(np.mean(v)) for k, v in random_omega.items()}
    return x, y, fit_split, r_bar, omega_bar


def _study_halfwidths(
    rng: np.random.Generator,
    checked: list[CheckedSplit],
    pooled_mean: dict[int, np.ndarray],
    pool_stacks: dict[int, list[np.ndarray]],
    n_draws: int,
    n_classes: int,
    delta: float,
    lam: float,
    gamma: float,
    kappa: float,
) -> dict[str, float]:
    """One simulated study's 95% interval half-width for each part."""
    x, y, fit_split, r_bar, omega_bar = _one_study(
        rng,
        checked,
        pooled_mean,
        pool_stacks,
        n_draws,
        n_classes,
        delta,
        lam,
        gamma,
        kappa,
    )
    w = draw_weights(fit_split, n_draws, N_STUDY_REPLICATES, rng)
    beta = estimate(y, x, fit_split, w)
    pts = parts(beta, r_bar[5], r_bar[10], omega_bar[5], omega_bar[10])
    out: dict[str, float] = {}
    for name, dist in pts.items():
        lo, hi = confidence_interval(dist)
        out[name] = (hi - lo) / 2.0
    return out


def _median_halfwidths(
    checked: list[CheckedSplit],
    pooled_mean: dict[int, np.ndarray],
    pool_stacks: dict[int, list[np.ndarray]],
    n_classes: int,
    n_draws: int,
    delta: float,
    lam: float,
    gamma: float,
    kappa: float,
    seed: int,
) -> dict[str, float]:
    """N_SIM simulated studies' median 95% interval half-width per part."""
    rng = np.random.default_rng(seed)
    halfwidths: dict[str, list[float]] = {"C": [], "S": [], "P": []}
    for _ in range(N_SIM):
        hw = _study_halfwidths(
            rng,
            checked,
            pooled_mean,
            pool_stacks,
            n_draws,
            n_classes,
            delta,
            lam,
            gamma,
            kappa,
        )
        for name, value in hw.items():
            halfwidths[name].append(value)
    return {name: float(np.median(values)) for name, values in halfwidths.items()}


def _scenario_halfwidths(
    checked: list[CheckedSplit],
    pooled_mean: dict[int, np.ndarray],
    pool_stacks: dict[int, list[np.ndarray]],
    n_classes: int,
    n_draws: int,
    d_idx: int,
    kappa: float,
) -> dict[str, dict[str, float]]:
    """Both scenarios' median half-widths for one draw count and dispersion."""
    scenarios = {
        "zero": (0.0, 0.0, 0.0),
        "coverage_dominant": (SCENARIO_C, SCENARIO_S, SCENARIO_P),
    }
    out: dict[str, dict[str, float]] = {}
    for scenario, (c_eff, s_eff, p_eff) in scenarios.items():
        seed = 20260915 + 7919 * d_idx + int(kappa * 10) + hash(scenario) % 97
        out[scenario] = _median_halfwidths(
            checked,
            pooled_mean,
            pool_stacks,
            n_classes,
            n_draws,
            c_eff,
            s_eff,
            p_eff,
            kappa,
            seed,
        )
    return out


def select_draws(
    checked: list[CheckedSplit],
    pooled_mean: dict[int, np.ndarray],
    pool_stacks: dict[int, list[np.ndarray]],
    n_classes: int,
) -> tuple[dict[str, Any], int | None]:
    """Simulated median half-widths per draw count, dispersion, and scenario."""
    results: dict[str, Any] = {}
    selected: int | None = None
    for d_idx, n_draws in enumerate(DRAW_COUNTS):
        by_dispersion = {
            str(kappa): _scenario_halfwidths(
                checked, pooled_mean, pool_stacks, n_classes, n_draws, d_idx, kappa
            )
            for kappa in _DISPERSIONS
        }
        results[str(n_draws)] = by_dispersion
        base = by_dispersion["1.0"]
        if selected is None and all(
            v <= HALFWIDTH_TOL_PP
            for scenario in base.values()
            for v in scenario.values()
        ):
            selected = n_draws
    return results, selected
