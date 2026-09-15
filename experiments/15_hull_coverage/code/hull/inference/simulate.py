"""Precision simulation mechanics: one simulated study, its interval half-widths, and draw-count selection.

A simulated study resamples checked census draws within split and reuses their
achieved validation coverage, hull residual, and similarity for all classes.
Class recall is the exp-5 class mean for the patient count, plus a training-draw
deviation resampled from a stored exp-5 fit of the same split, class, and patient
count, plus the imposed effects, plus that fit's own test-patient replicate noise.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.analysis.inference.gates import confidence_interval

from hull import DRAW_COUNTS, N_SPLITS
from hull.inference import (
    DISPERSION_SENSITIVITY,
    HALFWIDTH_TOL_PP,
    HULL_DOMINANT_PARTS,
    N_SIM,
    N_STUDY_REPLICATES,
    PATIENT_COUNTS,
)
from hull.inference.model import draw_weights, estimate, parts, random_means

__all__ = [
    "CheckedSplit",
    "Templates",
    "Slopes",
    "scenario_slopes",
    "study_halfwidths",
    "select_draws",
]


class CheckedSplit(NamedTuple):
    """One split's checked census draws.

    ``x`` is (D, 2, C, 3) validation r, h, omega at G=5 (index 0) and G=10
    (index 1); ``random`` is (D, 2, C), True for random-cell cohorts.
    """

    x: np.ndarray
    random: np.ndarray


class Templates(NamedTuple):
    """Stored exp-5 recall by patient count: class means (C,) and per-split fit stacks (F, C, R)."""

    mean: dict[int, np.ndarray]
    stacks: dict[int, list[np.ndarray]]


class Slopes(NamedTuple):
    """Imposed model coefficients for r, h, omega, and z."""

    delta: float
    eta: float
    lam: float
    gamma: float


def scenario_slopes(checked: list[CheckedSplit], target: dict[str, float]) -> Slopes:
    """Convert target parts (pp) into slopes using the checked random-cell differences."""
    x = np.concatenate([c.x for c in checked])
    rnd = np.concatenate([c.random for c in checked])
    means = random_means(x, rnd)
    lo, hi = means[5], means[10]
    return Slopes(
        delta=target["C"] / (hi.r - lo.r),
        eta=target["H"] / (hi.h - lo.h),
        lam=target["S"] / (hi.omega - lo.omega) if target["S"] else 0.0,
        gamma=target["P"],
    )


def _split_recall(
    rng: np.random.Generator,
    xs: np.ndarray,
    templates: Templates,
    split_idx: int,
    slopes: Slopes,
    kappa: float,
) -> np.ndarray:
    """(D, 2, C, R) simulated recall for one split's resampled draws."""
    n_draws, _, n_classes, _ = xs.shape
    y = np.empty((n_draws, 2, n_classes, N_STUDY_REPLICATES))
    for gi, g in enumerate(PATIENT_COUNTS):
        pool = templates.stacks[g][split_idx]
        chosen = rng.integers(0, pool.shape[0], size=(n_draws, n_classes))
        rec = pool[chosen, np.arange(n_classes)[None, :], :N_STUDY_REPLICATES]
        cell = xs[:, gi]
        imposed = slopes.delta * cell[..., 0] + slopes.eta * cell[..., 1]
        imposed = imposed + slopes.lam * cell[..., 2] + slopes.gamma * gi
        base = templates.mean[g] + kappa * (rec[..., 0] - templates.mean[g]) + imposed
        y[:, gi] = base[..., None] + rec - rec[..., :1]
    return y


def _one_study(
    rng: np.random.Generator,
    checked: list[CheckedSplit],
    templates: Templates,
    n_draws: int,
    slopes: Slopes,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One simulated study's (B, 2, C, 4) predictors, (B, 2, C, R) recall, and (B, 2, C) random mask."""
    xs_all, ys, rnds = [], [], []
    for s in range(N_SPLITS):
        slots = rng.integers(0, len(checked[s].x), size=n_draws)
        xs = checked[s].x[slots]
        ys.append(_split_recall(rng, xs, templates, s, slopes, kappa))
        z = np.broadcast_to(
            np.array([0.0, 1.0])[None, :, None, None], (*xs.shape[:3], 1)
        )
        xs_all.append(np.concatenate([xs, z], axis=-1))
        rnds.append(checked[s].random[slots])
    return np.concatenate(xs_all), np.concatenate(ys), np.concatenate(rnds)


def study_halfwidths(
    rng: np.random.Generator,
    checked: list[CheckedSplit],
    templates: Templates,
    n_draws: int,
    slopes: Slopes,
    kappa: float,
) -> dict[str, float]:
    """One simulated study's 95% interval half-width for each part."""
    x, y, rnd = _one_study(rng, checked, templates, n_draws, slopes, kappa)
    block_split = np.repeat(np.arange(N_SPLITS), n_draws)
    w = draw_weights(block_split, n_draws, N_STUDY_REPLICATES, rng)
    pts = parts(estimate(y, x, block_split, w), random_means(x, rnd))
    out: dict[str, float] = {}
    for name, dist in pts.items():
        lo, hi = confidence_interval(dist)
        out[name] = (hi - lo) / 2.0
    return out


def _median_halfwidths(
    checked: list[CheckedSplit],
    templates: Templates,
    n_draws: int,
    slopes: Slopes,
    kappa: float,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    runs = [
        study_halfwidths(rng, checked, templates, n_draws, slopes, kappa)
        for _ in range(N_SIM)
    ]
    return {name: float(np.median([r[name] for r in runs])) for name in runs[0]}


def _draw_count_results(
    checked: list[CheckedSplit],
    templates: Templates,
    scenarios: dict[str, Slopes],
    d_idx: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Median half-widths for one draw count, by dispersion and scenario."""
    n_draws = DRAW_COUNTS[d_idx]
    return {
        str(kappa): {
            name: _median_halfwidths(
                checked,
                templates,
                n_draws,
                slopes,
                kappa,
                20260921 + 7919 * d_idx + 101 * s_idx + int(kappa * 10),
            )
            for s_idx, (name, slopes) in enumerate(scenarios.items())
        }
        for kappa in (1.0, DISPERSION_SENSITIVITY)
    }


def select_draws(
    checked: list[CheckedSplit], templates: Templates
) -> tuple[dict[str, Any], int | None]:
    """Median half-widths per draw count, dispersion, and scenario; the smallest qualifying draw count."""
    scenarios = {
        "zero": Slopes(0.0, 0.0, 0.0, 0.0),
        "hull_dominant": scenario_slopes(checked, HULL_DOMINANT_PARTS),
    }
    results: dict[str, Any] = {}
    selected: int | None = None
    for d_idx, n_draws in enumerate(DRAW_COUNTS):
        results[str(n_draws)] = _draw_count_results(
            checked, templates, scenarios, d_idx
        )
        base = results[str(n_draws)]["1.0"]
        if selected is None and all(
            v <= HALFWIDTH_TOL_PP for sc in base.values() for v in sc.values()
        ):
            selected = n_draws
    return results, selected
