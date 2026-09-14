"""Precision-simulation templates and the max-t / shifted-percentile success rules."""

from __future__ import annotations

import numpy as np

from similarity import EFFECT_PP, THRESHOLD_PP

__all__ = ["PRIMARY_NAMES", "build_templates", "simulate"]

PRIMARY_NAMES = ("coverage_low", "coverage_high", "similarity_good", "similarity_poor")


def _center(matrix: np.ndarray, n_splits: int, n_draws: int) -> np.ndarray:
    """Reshape (F, R) to (S, H, R) and subtract the observed historical mean contrast."""
    reshaped = matrix.reshape(n_splits, n_draws, matrix.shape[-1])
    return reshaped - float(reshaped[:, :, 0].mean())


def _permutations(
    n_splits: int, n_draws: int, seed: int
) -> dict[str, list[np.ndarray]]:
    """Fixed, per-contrast, per-split permutation of the historical draws."""
    return {
        name: [
            np.random.default_rng(seed + 1000 * i + s).permutation(n_draws)
            for s in range(n_splits)
        ]
        for i, name in enumerate(PRIMARY_NAMES)
    }


def _scale(template: np.ndarray, kappa: float) -> np.ndarray:
    """Scale draw deviations around the per-split draw mean by kappa."""
    split_mean = template.mean(axis=1, keepdims=True)
    return split_mean + kappa * (template - split_mean)


def build_templates(
    t_sel_raw: np.ndarray, t_g_raw: np.ndarray, seed: int, kappa: float
) -> dict[str, np.ndarray]:
    """The five named (S, H, R) templates for one dispersion setting."""
    n_splits, n_draws = t_sel_raw.shape[:2]
    t_sel = _center(t_sel_raw, n_splits, n_draws)
    t_g = _center(t_g_raw, n_splits, n_draws)
    perms = _permutations(n_splits, n_draws, seed)

    scaled_sel = _scale(t_sel, kappa)
    out: dict[str, np.ndarray] = {"delta_g": _scale(t_g, kappa)}
    for name in PRIMARY_NAMES:
        perm = perms[name]
        out[name] = np.stack([scaled_sel[s, perm[s], :] for s in range(n_splits)])
    return out


def _study_contrast(
    template: np.ndarray, counts: list[np.ndarray], draws: int, world_column: int
) -> np.ndarray:
    """Center replicate fluctuations on one sampled study's patient-and-draw mean."""
    world = float(
        np.mean(
            [
                (count[0] / draws) @ template[s, :, world_column]
                for s, count in enumerate(counts)
            ]
        )
    )
    contrast = np.mean(
        [
            np.einsum("rh,hr->r", count, template[s]) / draws
            for s, count in enumerate(counts)
        ],
        axis=0,
    )
    # The world already contains the observed training-draw fluctuation.
    # Shift only the centered replicate errors; retain the fixed test-patient noise.
    return world + (contrast - contrast[0])


def _one_study(
    rng: np.random.Generator,
    templates: dict[str, np.ndarray],
    draws: int,
    n_replicates: int,
) -> dict[str, np.ndarray]:
    """One simulated study's base (zero-effect) replicate contrast per named template."""
    n_splits, n_draws_hist = next(iter(templates.values())).shape[:2]
    uniform = np.full(n_draws_hist, 1.0 / n_draws_hist)
    n_h = [rng.multinomial(draws, uniform) for _ in range(n_splits)]
    r_w = int(rng.integers(1, n_replicates))
    counts = [
        np.vstack(
            [n_h[s], rng.multinomial(draws, n_h[s] / draws, size=n_replicates - 1)]
        )
        for s in range(n_splits)
    ]  # each (R, H); row 0 is the fixed observed composition

    return {
        name: _study_contrast(template, counts, draws, r_w)
        for name, template in templates.items()
    }


def _max_t_critical(base: dict[str, np.ndarray]) -> float:
    """95th percentile of the max |t| over every named contrast's replicates; NaN if undefined."""
    names = list(base)
    e = {n: float(base[n][0]) for n in names}
    se = {n: float(np.std(base[n][1:], ddof=1)) for n in names}
    if any(not np.isfinite(se[n]) or se[n] == 0 for n in names):
        return float("nan")
    t = np.stack([np.abs((base[n][1:] - e[n]) / se[n]) for n in names])
    return float(np.percentile(t.max(axis=0), 95))


def _primary_success(base: dict[str, np.ndarray]) -> dict[str, tuple[bool, bool]]:
    """Max-t simultaneous-interval success for the four primary contrasts (report Rules)."""
    names = list(base)
    c = _max_t_critical(base)
    if np.isnan(c):
        return {n: (False, False) for n in names}
    e = {n: float(base[n][0]) for n in names}
    se = {n: float(np.std(base[n][1:], ddof=1)) for n in names}
    out: dict[str, tuple[bool, bool]] = {}
    for n in names:
        lo, hi = e[n] - c * se[n], e[n] + c * se[n]
        zero_ok = lo >= -THRESHOLD_PP and hi <= THRESHOLD_PP
        two_ok = (e[n] + EFFECT_PP - c * se[n]) > THRESHOLD_PP
        out[n] = (zero_ok, two_ok)
    return out


def _breadth_success(base: np.ndarray) -> tuple[bool, bool]:
    """Shifted-percentile 95% interval success for Delta_G (report Rules)."""
    reps = base[1:]
    if len(reps) < 2 or not np.all(np.isfinite(reps)):
        return False, False
    e = float(base[0])
    q_lo, q_hi = np.percentile(reps, [2.5, 97.5])
    lo, hi = 2.0 * e - q_hi, 2.0 * e - q_lo
    zero_ok = lo >= -THRESHOLD_PP and hi <= THRESHOLD_PP
    two_ok = (lo + EFFECT_PP) > THRESHOLD_PP
    return zero_ok, two_ok


def _mc_se(p: float, n_sim: int) -> float:
    return float(np.sqrt(p * (1.0 - p) / n_sim))


def simulate(
    templates: dict[str, np.ndarray],
    draws: int,
    n_sim: int,
    seed: int,
    n_replicates: int,
) -> dict[str, dict[str, float]]:
    """N_SIM simulated studies' success rates, per contrast and scenario."""
    rng = np.random.default_rng(seed)
    hits = {name: {"zero": 0, "two": 0} for name in templates}
    for _ in range(n_sim):
        base = _one_study(rng, templates, draws, n_replicates)
        outcomes = _primary_success({n: base[n] for n in PRIMARY_NAMES})
        outcomes["delta_g"] = _breadth_success(base["delta_g"])
        for name, (zero_ok, two_ok) in outcomes.items():
            hits[name]["zero"] += int(zero_ok)
            hits[name]["two"] += int(two_ok)
    return {
        name: {
            "zero": v["zero"] / n_sim,
            "two": v["two"] / n_sim,
            "zero_mc_se": _mc_se(v["zero"] / n_sim, n_sim),
            "two_mc_se": _mc_se(v["two"] / n_sim, n_sim),
        }
        for name, v in hits.items()
    }
