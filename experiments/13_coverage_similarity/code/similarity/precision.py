"""Precision simulation: choose the fresh-draw count before any training (report App. B)."""

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

from sites.recall import allocation_dirs, ctx_list, grid_dirs, perm_list

from neighbours.accuracy import recall_stack

from coverage_redundancy.rows import _guard_point_accuracy

from similarity import (
    DRAW_COUNTS,
    N_REPLICATES,
    N_SIM,
    N_SPLITS,
    POWER,
    SEARCH_BASE_SEED,
)
from similarity import exp5_config, exp12_config
from similarity.simulate import build_templates, simulate
from similarity.weights import weighted_contexts

__all__ = ["run_precision", "load_precision"]


def _class_mean_matrix(
    dirs: list[Path], ctx_l: list[Any], perms: list[np.ndarray], n_classes: int
) -> np.ndarray:
    """Per-fit, per-replicate accuracy (%) over all classes: (F, R)."""
    stack = recall_stack(dirs, ctx_l, perms, n_classes)
    return stack.mean(axis=1) * 100.0


def _read_analysis(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        (output_root(config) / "data" / "analysis.json").read_text(encoding="utf-8")
    )


def _guard_templates(
    config: dict[str, Any],
    dispersed: np.ndarray,
    random_: np.ndarray,
    g10_m16: np.ndarray,
) -> None:
    """Guard the observed point accuracies against exp-12's and exp-5's own analyses."""
    exp12_analysis = _read_analysis(exp12_config(config))
    exp5_analysis = _read_analysis(exp5_config(config))
    _guard_point_accuracy(
        float(dispersed[:, 0].mean()),
        float(exp12_analysis["subgroups"]["all"]["accuracy"]["dispersed"]["point"]),
        "exp-12 dispersed",
    )
    _guard_point_accuracy(
        float(random_[:, 0].mean()),
        float(exp12_analysis["subgroups"]["all"]["accuracy"]["random"]["point"]),
        "exp-12 random",
    )
    _guard_point_accuracy(
        float(g10_m16[:, 0].mean()),
        float(exp5_analysis["cell_accuracies"]["G10_m16"]["point"]),
        "Grid cell G10_m16",
    )


def _load_templates(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    """The raw (S, H, R) selection and breadth templates, guarded against their sources."""
    canonical_names = canonical_class_names(config)
    n_classes = len(canonical_names)
    ctx_l = ctx_list(weighted_contexts(config, N_REPLICATES, SEARCH_BASE_SEED))
    perms = perm_list(config, canonical_names)

    paths12 = {
        s: split_paths(ensure_dirs(exp12_config(config)), s) for s in range(N_SPLITS)
    }
    paths5 = {
        s: split_paths(ensure_dirs(exp5_config(config)), s) for s in range(N_SPLITS)
    }
    dispersed = _class_mean_matrix(
        allocation_dirs(paths12, "dispersed"), ctx_l, perms, n_classes
    )
    random_ = _class_mean_matrix(
        allocation_dirs(paths12, "random"), ctx_l, perms, n_classes
    )
    g10_m16 = _class_mean_matrix(grid_dirs(paths5, 10, 16), ctx_l, perms, n_classes)
    _guard_templates(config, dispersed, random_, g10_m16)

    n_replicates = dispersed.shape[1]
    t_sel = (dispersed - random_).reshape(N_SPLITS, _HIST_DRAWS, n_replicates)
    t_g = (g10_m16 - dispersed).reshape(N_SPLITS, _HIST_DRAWS, n_replicates)
    return t_sel, t_g, n_replicates


def _select_draws(
    t_sel: np.ndarray, t_g: np.ndarray, n_replicates: int
) -> tuple[dict[str, Any], int | None]:
    """Simulated success rates per draw count and dispersion, and the smallest qualifying D."""
    results: dict[str, Any] = {}
    selected: int | None = None
    for d_idx, draws in enumerate(DRAW_COUNTS):
        by_kappa: dict[str, Any] = {}
        for kappa in (1.0, 1.5):
            seed = SEARCH_BASE_SEED + 7919 * d_idx + int(kappa * 10)
            templates = build_templates(t_sel, t_g, seed, kappa)
            by_kappa[str(kappa)] = simulate(templates, draws, N_SIM, seed, n_replicates)
        results[str(draws)] = by_kappa
        base_rates = by_kappa["1.0"]
        if selected is None and all(
            v["zero"] >= POWER and v["two"] >= POWER for v in base_rates.values()
        ):
            selected = draws
    return results, selected


def _write_precision(
    config: dict[str, Any], results: dict[str, Any], selected: int | None
) -> Path:
    out_p = output_root(config) / "data" / "precision.json"
    write_json(
        out_p,
        {
            "draw_counts": list(DRAW_COUNTS),
            "power": POWER,
            "n_sim": N_SIM,
            "selected_draws": selected,
            "results": results,
        },
    )
    sign_file(out_p)
    return out_p


def run_precision(config: dict[str, Any]) -> Path:
    """Simulate the primary and breadth contrasts; select or fail the fresh-draw count.

    Writes before raising so a failing simulation is still inspectable, and
    raises when no draw count qualifies so the downstream census never runs.
    """
    t_sel, t_g, n_replicates = _load_templates(config)
    results, selected = _select_draws(t_sel, t_g, n_replicates)
    out_p = _write_precision(config, results, selected)
    if selected is None:
        raise RuntimeError(
            "No draw count of 20, 40, or 60 reached 80% simulated success in both "
            "scenarios; stopping with a precision limitation."
        )
    return out_p


def load_precision(config: dict[str, Any]) -> dict[str, Any]:
    """Load and verify the signed precision simulation."""
    precision_p = output_root(config) / "data" / "precision.json"
    verify_signed_file(precision_p)
    return json.loads(precision_p.read_text(encoding="utf-8"))
