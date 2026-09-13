"""Descriptive secondary analysis: coverage gain within the focal cell vs other cells."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from imbalance_benchmark.analysis.query import load_test_identity

from breadth import N_DRAWS, N_SPLITS, exp2_split_paths
from breadth.analyze.diagnostics import _mean_leaves

from neighbours.accuracy import PathsBySplit
from neighbours.census import load_allocations
from neighbours.secondary import _class_pairs, _draw_pairs, _group_rows

__all__ = ["focal_strata"]

_ALLOCATIONS = ("concentrated", "random")


def _strata_indices(
    case_ids: np.ndarray, focal_cell: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Positional indices of the focal-cell and other-cells test patients."""
    focal_set = set(focal_cell)
    is_focal = np.array([case in focal_set for case in case_ids])
    return np.flatnonzero(is_focal), np.flatnonzero(~is_focal)


def _accumulate_class(
    pairs: dict[str, pd.DataFrame],
    c_idx: int,
    focal_cell: list[str],
    strata_rows: tuple[list[Any], list[Any]],
) -> None:
    """Fold one (split, draw, class) fit into the running focal/other rows."""
    pairs_c = _class_pairs(pairs["concentrated"], c_idx)
    pairs_r = _class_pairs(pairs["random"], c_idx)
    if pairs_c.empty:
        return
    groups = _strata_indices(pairs_c["case_id"].to_numpy(), focal_cell)
    rows = _group_rows(pairs_c, pairs_r, list(groups), low="concentrated")
    for bucket, row in zip(strata_rows, rows):
        bucket.append(row)


def _accumulate_split(
    config: dict[str, Any],
    paths: PathsBySplit,
    s: int,
    class_names: list[str],
    cmap: dict[str, int],
    allocations: dict[str, Any],
    strata_rows: tuple[list[Any], list[Any]],
) -> None:
    """Fold every draw of one split into the running focal/other rows."""
    manifest = exp2_split_paths(config, s)["data"] / "manifest.csv"
    case_ids = load_test_identity(manifest, is_mil=False)["case_id"].to_numpy()
    split_allocations = allocations[str(s)]["allocations"]
    for d in range(N_DRAWS):
        pairs = _draw_pairs(paths, s, d, case_ids, _ALLOCATIONS)
        for c_name in class_names:
            focal_cell = cast(
                list[str], split_allocations[str(d)][c_name]["focal_cell"]
            )
            _accumulate_class(pairs, cmap[c_name], focal_cell, strata_rows)


def focal_strata(
    config: dict[str, Any], paths: PathsBySplit, class_names: list[str]
) -> dict[str, Any]:
    """Coverage gain within the focal cell and within the other cells (descriptive)."""
    cmap = {name: i for i, name in enumerate(class_names)}
    allocations = load_allocations(config)
    strata_rows: tuple[list[Any], list[Any]] = ([], [])
    for s in range(N_SPLITS):
        _accumulate_split(config, paths, s, class_names, cmap, allocations, strata_rows)
    return {
        name: _mean_leaves(bucket)
        for name, bucket in zip(("focal", "other"), strata_rows)
    }
