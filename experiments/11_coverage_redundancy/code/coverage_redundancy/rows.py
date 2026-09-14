"""Grid and composition cohort rows: predictors, accuracy matrices, and guards."""

from __future__ import annotations

import json
from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.common import output_root

from breadth import GRID_CELLS

from sites.recall import PathsBySplit, allocation_dirs, grid_dirs

from neighbours.accuracy import recall_stack

from coverage_redundancy import (
    COMPOSITION_ALLOCATIONS,
    N_DRAWS,
    exp5_config,
    exp10_config,
)

__all__ = [
    "GridRows",
    "CompositionRows",
    "cohort_index",
    "grid_rows",
    "composition_rows",
]

_ACCURACY_TOLERANCE = 1e-6


class GridRows(NamedTuple):
    """The 135 random cohorts' predictors and per-replicate accuracy (rows, R)."""

    split: np.ndarray
    log_neff: np.ndarray
    log_neff_omega: np.ndarray
    r: np.ndarray
    log_g: np.ndarray
    accuracy: np.ndarray


class CompositionRows(NamedTuple):
    """The 45 composition cohorts' predictors and per-replicate accuracy (rows, R)."""

    split: np.ndarray
    log_neff_omega: np.ndarray
    r: np.ndarray
    allocation: np.ndarray
    accuracy: np.ndarray


def cohort_index(quantities: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Map (source, split, g, m, allocation, draw) to its census record."""
    return {
        (c["source"], c["split"], c["g"], c["m"], c["allocation"], c["draw"]): c
        for c in quantities["cohorts"]
    }


def _guard_point_accuracy(actual: float, expected: float, what: str) -> None:
    """Raise unless a reconstructed point accuracy reproduces its stored source."""
    if abs(actual - expected) > _ACCURACY_TOLERANCE:
        raise ValueError(f"{what} accuracy {actual} does not reproduce {expected}")


def _read_analysis(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        (output_root(config) / "data" / "analysis.json").read_text(encoding="utf-8")
    )


def _grid_cell_rows(
    stack: np.ndarray, g: int, m: int, index: dict[tuple[Any, ...], dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """One grid cell's split indices, [log_neff, log_neff_omega, r, log_g], accuracy."""
    class_mean = stack.mean(axis=1) * 100.0  # (F=15, R)
    splits, rows = [], []
    for f in range(class_mean.shape[0]):
        split_idx, draw_idx = divmod(f, N_DRAWS)
        record = index[("random", split_idx, g, m, None, draw_idx)]
        splits.append(split_idx)
        rows.append(
            [
                record["log_neff"],
                record["log_neff_omega"],
                record["r"],
                float(np.log(g)),
            ]
        )
    return np.array(splits), np.array(rows)


def grid_rows(
    config: dict[str, Any],
    paths5: PathsBySplit,
    ctx_l: list[Any],
    perms: list[np.ndarray],
    n_classes: int,
    index: dict[tuple[Any, ...], dict[str, Any]],
) -> GridRows:
    """The 135 random cohorts' predictors and accuracy matrix, guarded against exp-5."""
    exp5_analysis = _read_analysis(exp5_config(config))
    splits, rows, accs = [], [], []
    for g, m in GRID_CELLS:
        stack = recall_stack(grid_dirs(paths5, g, m), ctx_l, perms, n_classes)
        cell_splits, cell_rows = _grid_cell_rows(stack, g, m, index)
        class_mean = stack.mean(axis=1) * 100.0
        _guard_point_accuracy(
            float(class_mean[:, 0].mean()),
            float(exp5_analysis["cell_accuracies"][f"G{g}_m{m}"]["point"]),
            f"Grid cell G{g}_m{m}",
        )
        splits.append(cell_splits)
        rows.append(cell_rows)
        accs.append(class_mean)
    predictors = np.concatenate(rows)
    return GridRows(
        np.concatenate(splits),
        predictors[:, 0],
        predictors[:, 1],
        predictors[:, 2],
        predictors[:, 3],
        np.concatenate(accs),
    )


def _composition_allocation_rows(
    stack: np.ndarray, name: str, index: dict[tuple[Any, ...], dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """One allocation's split indices and [log_neff_omega, r] predictor rows."""
    class_mean = stack.mean(axis=1) * 100.0  # (F=15, R)
    splits, rows = [], []
    for f in range(class_mean.shape[0]):
        split_idx, draw_idx = divmod(f, N_DRAWS)
        record = index[("composition", split_idx, 20, 8, name, draw_idx)]
        splits.append(split_idx)
        rows.append([record["log_neff_omega"], record["r"]])
    return np.array(splits), np.array(rows)


def _composition_cell(
    comp_analysis: dict[str, Any],
    stack: np.ndarray,
    name: str,
    index: dict[tuple[Any, ...], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One allocation's guarded split indices, predictor rows, and accuracy matrix."""
    class_mean = stack.mean(axis=1) * 100.0  # (F=15, R)
    _guard_point_accuracy(
        float(class_mean[:, 0].mean()),
        float(comp_analysis["subgroups"]["all"]["accuracy"][name]["point"]),
        f"Composition allocation {name}",
    )
    cell_splits, cell_rows = _composition_allocation_rows(stack, name, index)
    return cell_splits, cell_rows, class_mean


def composition_rows(
    config: dict[str, Any],
    paths10: PathsBySplit,
    ctx_l: list[Any],
    perms: list[np.ndarray],
    n_classes: int,
    index: dict[tuple[Any, ...], dict[str, Any]],
) -> tuple[CompositionRows, dict[str, np.ndarray]]:
    """The 45 composition cohorts' predictors, guarded against exp-10's own analysis."""
    comp_analysis = _read_analysis(exp10_config(config))
    splits, rows, allocation, accs = [], [], [], []
    accuracy_by_allocation: dict[str, np.ndarray] = {}
    for name in COMPOSITION_ALLOCATIONS:
        stack = recall_stack(allocation_dirs(paths10, name), ctx_l, perms, n_classes)
        cell_splits, cell_rows, class_mean = _composition_cell(
            comp_analysis, stack, name, index
        )
        accuracy_by_allocation[name] = class_mean
        splits.append(cell_splits)
        rows.append(cell_rows)
        allocation.extend([name] * len(cell_splits))
        accs.append(class_mean)
    predictors = np.concatenate(rows)
    out = CompositionRows(
        np.concatenate(splits),
        predictors[:, 0],
        predictors[:, 1],
        np.array(allocation),
        np.concatenate(accs),
    )
    return out, accuracy_by_allocation
