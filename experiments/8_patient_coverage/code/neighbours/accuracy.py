"""Grid and allocation accuracy loading, the surface refit, and the paired contrasts."""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.common import ensure_dirs, split_paths

from breadth import GRID_CELLS, N_SPLITS, exp2_split_paths

from redundancy import exp5_config
from redundancy.estimator import cell_effective_support
from redundancy.surfaces import LN2

from sites.fitting import _class_correlations, _guard_against_exp6, _surface_bootstrap
from sites.recall import PathsBySplit, contexts, ctx_list, grid_dirs
from sites.recall import _recall_matrix
from sites.recall import allocation_dirs as _site_allocation_dirs
from sites.stages import load_census as load_site_census

from neighbours import ALLOCATIONS, DEEP_CELL, exp7_config

__all__ = ["Context", "SubgroupFit", "prepare", "fit"]

logger = logging.getLogger(__name__)

_BROAD_CELL = (20, 8)
_NEFF_DEEP_EXPECTED = 10.06
_NEFF_BROAD_EXPECTED = 36.58
_NEFF_LOG_TOLERANCE = 0.5


class Context(NamedTuple):
    """Everything the grid refit and the allocation fit both need."""

    class_names: list[str]
    site_classes: list[str]
    subgroup_idx: dict[str, np.ndarray]
    ctx_l: list[Any]
    paths5: PathsBySplit
    paths8: PathsBySplit
    rho: dict[str, np.ndarray]


class SubgroupFit(NamedTuple):
    """One subgroup's allocation accuracy, refitted surface, and contrasts."""

    a_deep: np.ndarray
    a_neighbours: np.ndarray
    a_random: np.ndarray
    points_deep: np.ndarray
    points_neighbours: np.ndarray
    points_random: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    neff_deep: float
    neff_broad: float
    b_n: np.ndarray
    b_ref: np.ndarray
    delta_c: np.ndarray


def recall_stack(
    dirs: list[Any], class_names: list[str], ctxs: list[Any]
) -> np.ndarray:
    """Per-fit, per-class recall distributions (F, C, R), each fit loaded once."""
    return np.stack([_recall_matrix(d, class_names, ctx) for d, ctx in zip(dirs, ctxs)])


def subgroup_distribution(
    stack: np.ndarray, class_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (R,) and per-fit observed (F,) accuracy (%) over a class subset."""
    per_fit_class_mean = stack[:, class_idx, :].mean(axis=1) * 100.0
    return per_fit_class_mean.mean(axis=0), per_fit_class_mean[:, 0]


def _subgroup_indices(
    class_names: list[str], site_classes: list[str]
) -> dict[str, np.ndarray]:
    all_idx = np.arange(len(class_names))
    site_idx = np.array([class_names.index(c) for c in site_classes])
    other_idx = np.array([i for i in all_idx if i not in set(site_idx.tolist())])
    return {"all": all_idx, "site": site_idx, "other": other_idx}


def prepare(config: dict[str, Any]) -> Context:
    """Load context, subgroup class sets, and class correlations."""
    class_names = list(load_freeze_meta(exp2_split_paths(config, 0))["class_names"])
    site_classes = list(load_site_census(exp7_config(config))["site_classes"])
    subgroup_idx = _subgroup_indices(class_names, site_classes)
    ctx_l = ctx_list(contexts(config))
    paths5 = {
        s: split_paths(ensure_dirs(exp5_config(config)), s) for s in range(N_SPLITS)
    }
    paths8 = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}

    rho_site, rho_all = _class_correlations(config, class_names, site_classes)
    rho = {
        "all": rho_all,
        "site": rho_site,
        "other": rho_all[subgroup_idx["other"]],
    }
    return Context(class_names, site_classes, subgroup_idx, ctx_l, paths5, paths8, rho)


def _grid_stacks(ctx: Context) -> dict[tuple[int, int], np.ndarray]:
    """Every grid cell's (F, C, R) recall stack, loaded once for all subgroups."""
    return {
        (g, m): recall_stack(grid_dirs(ctx.paths5, g, m), ctx.class_names, ctx.ctx_l)
        for g, m in GRID_CELLS
    }


def _allocation_stacks(
    ctx: Context, grid_stacks: dict[tuple[int, int], np.ndarray]
) -> dict[str, np.ndarray]:
    """The deep (reused from the grid) and broad allocations' (F, C, R) stacks."""
    stacks = {"deep": grid_stacks[DEEP_CELL]}
    for name in ALLOCATIONS:
        stacks[name] = recall_stack(
            _site_allocation_dirs(ctx.paths8, name), ctx.class_names, ctx.ctx_l
        )
    return stacks


def _guard_neff(neff_deep: float, neff_broad: float) -> None:
    """Log (not raise) if the all-class Neff drifts from the reported reference."""
    if (
        abs(np.log(neff_deep) - np.log(_NEFF_DEEP_EXPECTED)) > _NEFF_LOG_TOLERANCE
        or abs(np.log(neff_broad) - np.log(_NEFF_BROAD_EXPECTED)) > _NEFF_LOG_TOLERANCE
    ):
        logger.warning(
            "All-class Neff(deep)=%.3f, Neff(broad)=%.3f drift from the reported "
            "10.06 / 36.58",
            neff_deep,
            neff_broad,
        )
    else:
        logger.info(
            "All-class Neff(deep)=%.3f, Neff(broad)=%.3f (expected 10.06 / 36.58)",
            neff_deep,
            neff_broad,
        )


def _subgroup_fit(
    ctx: Context,
    subgroup: str,
    grid_stacks: dict[tuple[int, int], np.ndarray],
    allocation_stacks: dict[str, np.ndarray],
) -> SubgroupFit:
    idx = ctx.subgroup_idx[subgroup]
    rho = ctx.rho[subgroup]
    grid_dist = {
        cell: subgroup_distribution(grid_stacks[cell], idx)[0] for cell in GRID_CELLS
    }
    neff_grid = cell_effective_support(rho[np.newaxis, :], GRID_CELLS)[0]
    beta, gamma = _surface_bootstrap(grid_dist, neff_grid)

    neff_deep = float(cell_effective_support(rho[np.newaxis, :], [DEEP_CELL])[0, 0])
    neff_broad = float(cell_effective_support(rho[np.newaxis, :], [_BROAD_CELL])[0, 0])
    if subgroup == "all":
        _guard_neff(neff_deep, neff_broad)
    log_ratio = float(np.log(neff_broad / neff_deep))

    a_deep, points_deep = subgroup_distribution(allocation_stacks["deep"], idx)
    a_neighbours, points_neighbours = subgroup_distribution(
        allocation_stacks["neighbours"], idx
    )
    a_random, points_random = subgroup_distribution(allocation_stacks["random"], idx)

    explained = beta * log_ratio
    b_n = (a_neighbours - a_deep - explained) / 2.0
    b_ref = gamma * LN2
    delta_c = a_random - a_neighbours

    return SubgroupFit(
        a_deep,
        a_neighbours,
        a_random,
        points_deep,
        points_neighbours,
        points_random,
        beta,
        gamma,
        neff_deep,
        neff_broad,
        b_n,
        b_ref,
        delta_c,
    )


def fit(config: dict[str, Any], ctx: Context) -> dict[str, SubgroupFit]:
    """Grid and allocation accuracy, the surface refit, and the contrasts, per subgroup."""
    grid_stacks = _grid_stacks(ctx)
    all_idx = ctx.subgroup_idx["all"]
    grid_all_point = {
        cell: float(subgroup_distribution(grid_stacks[cell], all_idx)[0][0])
        for cell in GRID_CELLS
    }
    neff_all_grid = cell_effective_support(ctx.rho["all"][np.newaxis, :], GRID_CELLS)[0]
    _guard_against_exp6(config, grid_all_point, neff_all_grid)

    allocation_stacks = _allocation_stacks(ctx, grid_stacks)
    return {
        subgroup: _subgroup_fit(ctx, subgroup, grid_stacks, allocation_stacks)
        for subgroup in ctx.subgroup_idx
    }
