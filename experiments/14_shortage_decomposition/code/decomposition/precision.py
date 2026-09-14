"""Precision simulation: choose the fresh-draw count before any training (report App. "Precision simulation")."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import (
    ensure_dirs,
    output_root,
    sign_file,
    split_paths,
    verify_signed_file,
    write_json,
)

from breadth import N_DRAWS as _HIST_DRAWS
from breadth.analyze.canonical import canonical_class_names

from sites.recall import allocation_dirs, contexts, ctx_list, grid_dirs, perm_list

from neighbours.accuracy import recall_stack

from coverage_redundancy.rows import _guard_point_accuracy

from decomposition import (
    DRAW_COUNTS,
    HALFWIDTH_TOL_PP,
    N_DRAWS_DEFAULT,
    N_SPLITS,
    N_STUDY_REPLICATES,
)
from decomposition import exp5_config, exp12_config
from decomposition.census import load_allocations
from decomposition.simulate import CheckedSplit, select_draws

__all__ = ["run_precision", "load_precision"]


def _read_analysis(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        (output_root(config) / "data" / "analysis.json").read_text(encoding="utf-8")
    )


def _guard_templates(
    config: dict[str, Any],
    g5_m32: np.ndarray,
    g10_m16: np.ndarray,
    random_: np.ndarray,
    dispersed: np.ndarray,
) -> None:
    """Guard the observed point accuracies against exp-5's and exp-12's own analyses."""
    exp5_analysis = _read_analysis(exp5_config(config))
    exp12_analysis = _read_analysis(exp12_config(config))
    _guard_point_accuracy(
        float(g5_m32[:, :, 0].mean()),
        float(exp5_analysis["cell_accuracies"]["G5_m32"]["point"]),
        "Grid cell G5_m32",
    )
    _guard_point_accuracy(
        float(g10_m16[:, :, 0].mean()),
        float(exp5_analysis["cell_accuracies"]["G10_m16"]["point"]),
        "Grid cell G10_m16",
    )
    _guard_point_accuracy(
        float(random_[:, :, 0].mean()),
        float(exp12_analysis["subgroups"]["all"]["accuracy"]["random"]["point"]),
        "exp-12 random",
    )
    _guard_point_accuracy(
        float(dispersed[:, :, 0].mean()),
        float(exp12_analysis["subgroups"]["all"]["accuracy"]["dispersed"]["point"]),
        "exp-12 dispersed",
    )


def _load_templates(
    config: dict[str, Any],
) -> tuple[list[str], dict[int, np.ndarray], dict[int, list[np.ndarray]]]:
    """Per-class pooled class means and per-split stored-fit pools, by patient count."""
    canonical_names = canonical_class_names(config)
    n_classes = len(canonical_names)
    ctx_l = ctx_list(contexts(config))
    perms = perm_list(config, canonical_names)

    paths5 = {
        s: split_paths(ensure_dirs(exp5_config(config)), s) for s in range(N_SPLITS)
    }
    paths12 = {
        s: split_paths(ensure_dirs(exp12_config(config)), s) for s in range(N_SPLITS)
    }
    # In percent, to match analyze.py's model scale and exp-5/exp-12's own stored accuracies.
    g5_m32 = recall_stack(grid_dirs(paths5, 5, 32), ctx_l, perms, n_classes) * 100.0
    g10_m16 = recall_stack(grid_dirs(paths5, 10, 16), ctx_l, perms, n_classes) * 100.0
    random_ = (
        recall_stack(allocation_dirs(paths12, "random"), ctx_l, perms, n_classes)
        * 100.0
    )
    dispersed = (
        recall_stack(allocation_dirs(paths12, "dispersed"), ctx_l, perms, n_classes)
        * 100.0
    )
    _guard_templates(config, g5_m32, g10_m16, random_, dispersed)

    def _by_split(stack: np.ndarray) -> np.ndarray:
        return stack.reshape(N_SPLITS, _HIST_DRAWS, n_classes, -1)

    g5_s, g10_s, random_s, dispersed_s = (
        _by_split(g5_m32),
        _by_split(g10_m16),
        _by_split(random_),
        _by_split(dispersed),
    )
    pool_stacks = {
        5: [
            np.concatenate([g5_s[s], random_s[s], dispersed_s[s]], axis=0)
            for s in range(N_SPLITS)
        ],
        10: [g10_s[s] for s in range(N_SPLITS)],
    }
    pooled_mean = {
        5: np.concatenate(
            [g5_m32[:, :, 0], random_[:, :, 0], dispersed[:, :, 0]], axis=0
        ).mean(axis=0),
        10: g10_m16[:, :, 0].mean(axis=0),
    }
    return canonical_names, pooled_mean, pool_stacks


def _checked_splits(
    config: dict[str, Any], canonical_names: list[str]
) -> list[CheckedSplit]:
    """Every split's checked-draw achieved cells, ordered by canonical class."""
    out: list[CheckedSplit] = []
    for s in range(N_SPLITS):
        record = load_allocations(config, s)
        if record["draws"] != N_DRAWS_DEFAULT:
            raise RuntimeError(
                f"Census split {s} has {record['draws']} draws; the precision "
                f"simulation needs the {N_DRAWS_DEFAULT}-draw checked census."
            )
        by_draw_class = {(r["draw"], r["class"]): r for r in record["rows"]}
        r_val = np.empty((N_DRAWS_DEFAULT, len(canonical_names)))
        omega = np.empty_like(r_val)
        g = np.empty((N_DRAWS_DEFAULT, len(canonical_names)), dtype=np.int64)
        r_level = np.empty_like(g)
        for d in range(N_DRAWS_DEFAULT):
            for c_idx, c_name in enumerate(canonical_names):
                row = by_draw_class[(d, c_name)]
                r_val[d, c_idx] = row["r_val"]
                omega[d, c_idx] = row["omega"]
                g[d, c_idx] = row["g"]
                r_level[d, c_idx] = -1 if row["r_level"] is None else row["r_level"]
        out.append(CheckedSplit(r_val, omega, g, r_level))
    return out


def _write_precision(
    config: dict[str, Any], results: dict[str, Any], selected: int | None
) -> Path:
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
    return out_p


def run_precision(config: dict[str, Any]) -> Path:
    """Simulate the class-recall model; select or fail the fresh-draw count.

    Writes before raising so a failing simulation is still inspectable, and
    raises when no draw count qualifies so the downstream census re-run and
    fit array never start.
    """
    canonical_names, pooled_mean, pool_stacks = _load_templates(config)
    checked = _checked_splits(config, canonical_names)
    results, selected = select_draws(
        checked, pooled_mean, pool_stacks, len(canonical_names)
    )
    out_p = _write_precision(config, results, selected)
    if selected is None:
        raise RuntimeError(
            "No draw count of 20, 40, or 60 reached a median 95% half-width of "
            "1 point for C, S, and P in both scenarios; stopping with a precision "
            "limitation."
        )
    return out_p


def load_precision(config: dict[str, Any]) -> dict[str, Any]:
    """Load and verify the signed precision simulation."""
    precision_p = output_root(config) / "data" / "precision.json"
    verify_signed_file(precision_p)
    return json.loads(precision_p.read_text(encoding="utf-8"))
