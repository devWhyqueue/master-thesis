"""Secondary analysis 1: the selection gain's recovered share of the breadth-depth grid."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths

from breadth import N_SPLITS
from breadth.analyze.secondary import pack_estimate

from sites.recall import grid_dirs

from neighbours.accuracy import recall_stack, subgroup_distribution

from coverage_redundancy.rows import _guard_point_accuracy

from composition.analyze import Context

from shortage import N_DRAWS, exp5_config

__all__ = ["recovered_share", "log_split0_check"]

logger = logging.getLogger(__name__)


def _read_analysis(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        (output_root(config) / "data" / "analysis.json").read_text(encoding="utf-8")
    )


def _guard_grid_cells(
    exp5_analysis: dict[str, Any], points_20_8: np.ndarray, points_5_32: np.ndarray
) -> None:
    """Guard: the "all"-subgroup grid-cell points must reproduce exp-5's own analysis.json."""
    _guard_point_accuracy(
        float(points_20_8.mean()),
        float(exp5_analysis["cell_accuracies"]["G20_m8"]["point"]),
        "Grid cell G20_m8",
    )
    _guard_point_accuracy(
        float(points_5_32.mean()),
        float(exp5_analysis["cell_accuracies"]["G5_m32"]["point"]),
        "Grid cell G5_m32",
    )


def recovered_share(
    config: dict[str, Any], ctx: Context, delta_sel_by_subgroup: dict[str, np.ndarray]
) -> tuple[dict[str, Any], np.ndarray]:
    """Recovered share delta_sel / (A_20,8 - A_5,32), per replicate and subgroup."""
    paths5 = {
        s: split_paths(ensure_dirs(exp5_config(config)), s) for s in range(N_SPLITS)
    }
    n_classes = len(ctx.class_names)
    stack_20_8 = recall_stack(grid_dirs(paths5, 20, 8), ctx.ctx_l, ctx.perms, n_classes)
    stack_5_32 = recall_stack(grid_dirs(paths5, 5, 32), ctx.ctx_l, ctx.perms, n_classes)

    out: dict[str, Any] = {}
    points_5_32_all = None
    for subgroup, idx in ctx.subgroup_idx.items():
        a_20_8, points_20_8 = subgroup_distribution(stack_20_8, idx)
        a_5_32, points_5_32 = subgroup_distribution(stack_5_32, idx)
        if subgroup == "all":
            _guard_grid_cells(
                _read_analysis(exp5_config(config)), points_20_8, points_5_32
            )
            points_5_32_all = points_5_32
        out[subgroup] = pack_estimate(
            delta_sel_by_subgroup[subgroup] / (a_20_8 - a_5_32)
        )
    assert points_5_32_all is not None
    return out, points_5_32_all


def log_split0_check(
    random_points_all: np.ndarray, points_5_32_all: np.ndarray
) -> None:
    """Log (not guard) split-0 random-arm accuracy against exp-5's G5_m32 split-0 points."""
    logger.info(
        "Split-0 random-arm points %s vs exp-5 G5_m32 split-0 points %s (same patients)",
        random_points_all[:N_DRAWS].tolist(),
        points_5_32_all[:N_DRAWS].tolist(),
    )
