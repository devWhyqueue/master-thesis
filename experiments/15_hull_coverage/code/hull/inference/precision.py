"""Precision stage: choose the draw count before any classifier of this experiment is trained."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import (
    ensure_dirs,
    output_root,
    sign_file,
    split_paths,
    write_json,
)

from breadth.analyze.canonical import canonical_class_names

from sites import N_DRAWS as EXP5_DRAWS
from sites.recall import contexts, ctx_list, grid_dirs, perm_list

from neighbours.accuracy import recall_stack

from coverage_redundancy.rows import _guard_point_accuracy

from decomposition import exp5_config
from decomposition.census import load_allocations
from decomposition.precision import _read_analysis

from hull import DRAW_COUNTS, N_DRAWS_DEFAULT, N_SPLITS
from hull.inference import HALFWIDTH_TOL_PP, N_STUDY_REPLICATES, PATIENT_COUNTS
from hull.inference.simulate import CheckedSplit, Templates, select_draws

__all__ = ["run_precision"]

_EXP5_DEPTH = {5: 32, 10: 16}


def _load_templates(config: dict[str, Any], canonical_names: list[str]) -> Templates:
    """Stored exp-5 class recall at 5x32 and 10x16, guarded against exp-5's own cell accuracies."""
    n_classes = len(canonical_names)
    ctx_l, perms = ctx_list(contexts(config)), perm_list(config, canonical_names)
    exp5 = exp5_config(config)
    paths5 = {s: split_paths(ensure_dirs(exp5), s) for s in range(N_SPLITS)}
    cell_accuracies = _read_analysis(exp5)["cell_accuracies"]
    mean: dict[int, np.ndarray] = {}
    stacks: dict[int, list[np.ndarray]] = {}
    for g in PATIENT_COUNTS:
        label = f"G{g}_m{_EXP5_DEPTH[g]}"
        stack = (
            recall_stack(grid_dirs(paths5, g, _EXP5_DEPTH[g]), ctx_l, perms, n_classes)
            * 100.0
        )
        _guard_point_accuracy(
            float(stack[:, :, 0].mean()),
            float(cell_accuracies[label]["point"]),
            f"Grid cell {label}",
        )
        by_split = stack.reshape(N_SPLITS, EXP5_DRAWS, n_classes, -1)
        stacks[g] = [by_split[s] for s in range(N_SPLITS)]
        mean[g] = stack[:, :, 0].mean(axis=0)
    return Templates(mean, stacks)


def _checked_split(
    config: dict[str, Any], split_idx: int, canonical_names: list[str]
) -> CheckedSplit:
    """One split's checked census draws as arrays ordered by canonical class."""
    record = load_allocations(config, split_idx)
    if record["draws"] != N_DRAWS_DEFAULT:
        raise RuntimeError(
            f"Census split {split_idx} has {record['draws']} draws, expected {N_DRAWS_DEFAULT}"
        )
    shape = (N_DRAWS_DEFAULT, 2, len(canonical_names))
    x, random = np.full((*shape, 3), np.nan), np.zeros(shape, dtype=bool)
    for row in record["rows"]:
        if row["g"] not in PATIENT_COUNTS:
            continue
        pos = (
            row["draw"],
            PATIENT_COUNTS.index(row["g"]),
            canonical_names.index(row["class"]),
        )
        x[pos] = (row["r_val"], row["h_val"], row["omega"])
        random[pos] = row["mean_level"] is None
    if np.isnan(x).any():
        raise RuntimeError(f"Census split {split_idx} is missing cohorts")
    return CheckedSplit(x, random)


def run_precision(config: dict[str, Any]) -> Path:
    """Simulate the class-recall model and select the draw count; write, sign, then raise on failure."""
    canonical_names = canonical_class_names(config)
    templates = _load_templates(config, canonical_names)
    checked = [_checked_split(config, s, canonical_names) for s in range(N_SPLITS)]
    results, selected = select_draws(checked, templates)
    out_p = output_root(config) / "data" / "precision.json"
    write_json(
        out_p,
        {
            "draw_counts": list(DRAW_COUNTS),
            "halfwidth_tol_pp": HALFWIDTH_TOL_PP,
            "n_study_replicates": N_STUDY_REPLICATES,
            "selected_draws": selected,
            "results": results,
        },
    )
    sign_file(out_p)
    if selected is None:
        raise RuntimeError(
            "No draw count reached a median 95% half-width of 1 point for C, H, S, and P"
        )
    return out_p
