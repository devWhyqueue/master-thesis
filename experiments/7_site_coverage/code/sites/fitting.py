"""Grid and allocation accuracy loading, the surface refit, and the paired contrasts."""

from __future__ import annotations

import json
from typing import Any, NamedTuple

import numpy as np
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths

from breadth import GRID_CELLS, N_SPLITS, exp2_split_paths
from breadth.surface import fit_candidate_models

from redundancy import exp5_config
from redundancy.analyze import _clipped_split_mean, _load_correlations
from redundancy.estimator import cell_effective_support
from redundancy.surfaces import LN2

from sites import exp6_config
from sites.recall import (
    PathsBySplit,
    allocation_distribution,
    contexts,
    ctx_list,
    grid_dirs,
)
from sites.recall import allocation_dirs as _allocation_dirs
from sites.stages import load_census

__all__ = ["Context", "FitResult", "prepare", "fit"]

_EXP6_GUARD_TOLERANCE = 1e-6

Contrasts = tuple[
    np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray
]


class Context(NamedTuple):
    """Everything the grid refit and the new allocations' fit both need."""

    class_names: list[str]
    site_classes: list[str]
    site_idx: np.ndarray
    all_idx: np.ndarray
    ctx_l: list[Any]
    paths5: PathsBySplit
    paths7: PathsBySplit
    rho_site: np.ndarray
    rho_all: np.ndarray


class FitResult(NamedTuple):
    """The three allocations' accuracy distributions and the paired contrasts."""

    dists: dict[str, np.ndarray]
    all_dists: dict[str, np.ndarray]
    points: dict[str, np.ndarray]
    contrasts: Contrasts


def _class_correlations(
    config: dict[str, Any], class_names: list[str], site_classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Point full-feature ICC (Eq. correlation), restricted to site and all classes."""
    raw, exp6_class_names = _load_correlations(exp6_config(config))
    rho_full = _clipped_split_mean(raw["full"])[0]
    rho_by_class = dict(zip(exp6_class_names, rho_full))
    rho_site = np.array([rho_by_class[c] for c in site_classes])
    rho_all = np.array([rho_by_class[c] for c in class_names])
    return rho_site, rho_all


def prepare(config: dict[str, Any]) -> Context:
    """Load context and class correlations needed by every downstream stage."""
    class_names = list(load_freeze_meta(exp2_split_paths(config, 0))["class_names"])
    site_classes = load_census(config)["site_classes"]
    site_idx = np.array([class_names.index(c) for c in site_classes])
    all_idx = np.arange(len(class_names))
    ctx_l = ctx_list(contexts(config))
    paths5 = {
        s: split_paths(ensure_dirs(exp5_config(config)), s) for s in range(N_SPLITS)
    }
    paths7 = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    rho_site, rho_all = _class_correlations(config, class_names, site_classes)
    return Context(
        class_names,
        site_classes,
        site_idx,
        all_idx,
        ctx_l,
        paths5,
        paths7,
        rho_site,
        rho_all,
    )


def _guard_against_exp6(
    config: dict[str, Any],
    grid_all_point: dict[tuple[int, int], float],
    neff_all_grid: np.ndarray,
) -> None:
    """Guard: the all-class point fit must reproduce exp-6's own surface point b."""
    accs_all = np.array([grid_all_point[c] for c in GRID_CELLS])
    fit_result = fit_candidate_models(GRID_CELLS, accs_all, neff_all_grid)
    guard_b = fit_result["augmented_effective"]["gamma_e"] * LN2
    exp6_analysis_p = output_root(exp6_config(config)) / "data" / "analysis.json"
    exp6_analysis = json.loads(exp6_analysis_p.read_text(encoding="utf-8"))
    expected_b = exp6_analysis["surfaces"]["full"]["b"]["point"]
    if abs(guard_b - expected_b) > _EXP6_GUARD_TOLERANCE:
        raise ValueError(
            f"All-class site-coverage surface b={guard_b} does not reproduce "
            f"exp-6's b={expected_b}"
        )


def _grid_fit(
    config: dict[str, Any], ctx: Context
) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray]:
    """Site-class-restricted grid accuracy distributions and their Neff array."""
    grid_site_dist: dict[tuple[int, int], np.ndarray] = {}
    grid_all_point: dict[tuple[int, int], float] = {}
    for g, m in GRID_CELLS:
        dirs = grid_dirs(ctx.paths5, g, m)
        site_dist, _ = allocation_distribution(
            dirs, ctx.class_names, ctx.ctx_l, ctx.site_idx
        )
        all_dist, _ = allocation_distribution(
            dirs, ctx.class_names, ctx.ctx_l, ctx.all_idx
        )
        grid_site_dist[(g, m)] = site_dist
        grid_all_point[(g, m)] = float(all_dist[0])
    neff_site_grid = cell_effective_support(ctx.rho_site[np.newaxis, :], GRID_CELLS)[0]
    neff_all_grid = cell_effective_support(ctx.rho_all[np.newaxis, :], GRID_CELLS)[0]
    _guard_against_exp6(config, grid_all_point, neff_all_grid)
    return grid_site_dist, neff_site_grid


def _allocation_fit(
    ctx: Context,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Site-class and all-class accuracy of the three allocations, plus per-fit points."""
    dists: dict[str, np.ndarray] = {}
    all_dists: dict[str, np.ndarray] = {}
    points: dict[str, np.ndarray] = {}
    for name in ("deep", "broad5", "broad10"):
        dirs = _allocation_dirs(ctx.paths7, name)
        site_dist, site_points = allocation_distribution(
            dirs, ctx.class_names, ctx.ctx_l, ctx.site_idx
        )
        all_dist, _ = allocation_distribution(
            dirs, ctx.class_names, ctx.ctx_l, ctx.all_idx
        )
        dists[name], all_dists[name], points[name] = site_dist, all_dist, site_points
    return dists, all_dists, points


def _surface_bootstrap(
    grid_site_dist: dict[tuple[int, int], np.ndarray], neff_site_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Refit the site-class-restricted surface on every bootstrap replicate."""
    n_reps = len(next(iter(grid_site_dist.values())))
    beta, gamma = np.empty(n_reps), np.empty(n_reps)
    for b in range(n_reps):
        accs = np.array([grid_site_dist[c][b] for c in GRID_CELLS])
        fit_result = fit_candidate_models(GRID_CELLS, accs, neff_site_grid)
        beta[b] = fit_result["augmented_effective"]["beta"]
        gamma[b] = fit_result["augmented_effective"]["gamma_e"]
    return beta, gamma


def _contrasts(
    rho_site: np.ndarray,
    grid_site_dist: dict[tuple[int, int], np.ndarray],
    neff_site_grid: np.ndarray,
    allocation_dists: dict[str, np.ndarray],
) -> Contrasts:
    """Refit beta/gamma, then the paired site-gain and within-site contrasts."""
    beta_dist, gamma_dist = _surface_bootstrap(grid_site_dist, neff_site_grid)
    neff_deep = float(cell_effective_support(rho_site[np.newaxis, :], [(5, 32)])[0, 0])
    neff_broad = float(cell_effective_support(rho_site[np.newaxis, :], [(20, 8)])[0, 0])
    log_ratio = float(np.log(neff_broad / neff_deep))
    explained = beta_dist * log_ratio
    b_w = (allocation_dists["broad5"] - allocation_dists["deep"] - explained) / 2.0
    b_ref = gamma_dist * LN2
    delta_s = allocation_dists["broad10"] - allocation_dists["broad5"]
    return beta_dist, gamma_dist, neff_deep, neff_broad, b_w, b_ref, delta_s


def fit(config: dict[str, Any], ctx: Context) -> FitResult:
    """Grid and allocation accuracy, the surface refit, and the paired contrasts."""
    grid_site_dist, neff_site_grid = _grid_fit(config, ctx)
    dists, all_dists, points = _allocation_fit(ctx)
    contrasts = _contrasts(ctx.rho_site, grid_site_dist, neff_site_grid, dists)
    return FitResult(dists, all_dists, points, contrasts)
