"""Bootstrap distributions, contrasts, and support surface parameter extraction."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from decodability import exp2_split_paths
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.gates import confidence_interval
from imbalance_benchmark.analysis.query import read_run_record
from imbalance_benchmark.common import (
    ensure_dirs,
    output_root,
    split_paths,
    verify_signed_file,
)

from breadth import (
    BOOTSTRAP_SEED,
    BREADTH_LADDER,
    DEPTH_LADDER,
    GRID_CELLS,
    N_DRAWS,
    N_REPLICATES,
    N_SPLITS,
    draw_dir,
)
from breadth.surface import fit_candidate_models

__all__ = [
    "load_draw_record",
    "collect_cell_distributions",
    "compute_contrasts_and_surface",
]


def load_draw_record(
    config: dict[str, Any], split_index: int, g: int, m: int, draw_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test labels, predictions, and probabilities for one (split, cell, draw)."""
    paths = split_paths(ensure_dirs(config), split_index)
    d_dir = draw_dir(paths, g, m, draw_index)
    rec = read_run_record(
        d_dir, splits=("test",), array_fields=("labels", "preds", "probabilities")
    )
    if rec is None or "test" not in rec.get("splits", {}):
        raise RuntimeError(f"Missing run record at {d_dir}")
    t_data = rec["splits"]["test"]
    return (
        np.asarray(t_data["labels"]),
        np.asarray(t_data["preds"]),
        np.asarray(t_data["probabilities"]),
    )


def _pack_estimate(dist: np.ndarray) -> dict[str, float]:
    """Extract point estimate (index 0) and 95% CI from bootstrap distribution."""
    ci = confidence_interval(dist)
    return {
        "point": float(dist[0]),
        "ci_2_5": float(ci[0]),
        "ci_97_5": float(ci[1]),
    }


def _collect_cell_replicates(
    contexts: dict[int, BootstrapContext],
    config: dict[str, Any],
    n_classes: int,
) -> tuple[
    dict[tuple[int, int], list[list[np.ndarray]]], dict[tuple[int, int], list[float]]
]:
    """Sample bootstrap distributions for each split, cell, and draw."""
    cell_splits: dict[tuple[int, int], list[list[Any]]] = {
        c: [[None] * N_DRAWS for _ in range(N_SPLITS)] for c in GRID_CELLS
    }
    raw_points: dict[tuple[int, int], list[float]] = {c: [] for c in GRID_CELLS}

    for s_idx in range(N_SPLITS):
        ctx = contexts[s_idx]
        for g, m in GRID_CELLS:
            for d_idx in range(N_DRAWS):
                labels, preds, _ = load_draw_record(config, s_idx, g, m, d_idx)
                dist = (
                    ctx.ba_distribution(labels, preds[np.newaxis, :], n_classes) * 100.0
                )
                cell_splits[(g, m)][s_idx][d_idx] = dist
                raw_points[(g, m)].append(float(dist[0]))
    return cell_splits, raw_points


def collect_cell_distributions(
    config: dict[str, Any], n_classes: int
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], float]]:
    """Gather 3-split pooled bootstrap distributions and between-draw dispersions."""
    contexts = {
        i: BootstrapContext(
            exp2_split_paths(config, i),
            is_mil=False,
            n_replicates=N_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        for i in range(N_SPLITS)
    }
    cell_splits, raw_points = _collect_cell_replicates(contexts, config, n_classes)
    pooled_dists: dict[tuple[int, int], np.ndarray] = {}
    dispersions: dict[tuple[int, int], float] = {}

    for c in GRID_CELLS:
        split_means = [np.mean(cell_splits[c][s], axis=0) for s in range(N_SPLITS)]
        pooled_dists[c] = np.mean(split_means, axis=0)
        dispersions[c] = float(np.std(raw_points[c]))
    return pooled_dists, dispersions


def _fit_replicate_surface(
    cells: tuple[tuple[int, int], ...],
    neff_arr: np.ndarray,
    pooled_dists: dict[tuple[int, int], np.ndarray],
    b: int,
) -> dict[str, float]:
    """Fit candidate models on a single bootstrap replicate."""
    accs = np.array([pooled_dists[c][b] for c in cells])
    m = fit_candidate_models(cells, accs, neff_arr)
    return {
        "beta_n": m["single_models"]["log_n"]["beta"],
        "res_std_n": m["single_models"]["log_n"]["res_std"],
        "beta_g": m["single_models"]["log_g"]["beta"],
        "res_std_g": m["single_models"]["log_g"]["res_std"],
        "beta_neff": m["single_models"]["log_neff"]["beta"],
        "res_std_neff": m["single_models"]["log_neff"]["res_std"],
        "gamma_n": m["augmented_nominal"]["gamma_n"],
        "res_std_aug_nom": m["augmented_nominal"]["res_std"],
        "gamma_e": m["augmented_effective"]["gamma_e"],
        "res_std_aug_eff": m["augmented_effective"]["res_std"],
    }


def _run_surface_bootstrap(
    cells: tuple[tuple[int, int], ...],
    neff_arr: np.ndarray,
    pooled_dists: dict[tuple[int, int], np.ndarray],
) -> dict[str, dict[str, float]]:
    """Run surface fits across all bootstrap replicates and extract intervals."""
    n_reps = len(next(iter(pooled_dists.values())))
    param_keys = [
        "beta_n",
        "res_std_n",
        "beta_g",
        "res_std_g",
        "beta_neff",
        "res_std_neff",
        "gamma_n",
        "res_std_aug_nom",
        "gamma_e",
        "res_std_aug_eff",
    ]
    reps: dict[str, list[float]] = {k: [] for k in param_keys}
    for b in range(n_reps):
        row = _fit_replicate_surface(cells, neff_arr, pooled_dists, b)
        for k in param_keys:
            reps[k].append(row[k])
    return {k: _pack_estimate(np.array(vals)) for k, vals in reps.items()}


def _build_contrasts(pooled_dists: dict[tuple[int, int], np.ndarray]) -> dict[str, Any]:
    """Calculate and pack contrast bootstrap estimates."""
    delta_m = {g: pooled_dists[(g, 32)] - pooled_dists[(g, 8)] for g in BREADTH_LADDER}
    delta_g = {m: pooled_dists[(20, m)] - pooled_dists[(5, m)] for m in DEPTH_LADDER}
    x_dist = pooled_dists[(20, 8)] - pooled_dists[(5, 32)]
    return {
        "delta_m": {str(g): _pack_estimate(delta_m[g]) for g in BREADTH_LADDER},
        "delta_g": {str(m): _pack_estimate(delta_g[m]) for m in DEPTH_LADDER},
        "equal_budget_advantage_X": _pack_estimate(x_dist),
        "isobudget_160_series": {
            "G20_m8": _pack_estimate(pooled_dists[(20, 8)]),
            "G10_m16": _pack_estimate(pooled_dists[(10, 16)]),
            "G5_m32": _pack_estimate(pooled_dists[(5, 32)]),
        },
    }


def compute_contrasts_and_surface(
    config: dict[str, Any],
    pooled_dists: dict[tuple[int, int], np.ndarray],
    draw_dispersions: dict[tuple[int, int], float],
) -> dict[str, Any]:
    """Compute contrasts and bootstrap distributions of support surface fits."""
    preflight_p = output_root(config) / "data" / "preflight.json"
    verify_signed_file(preflight_p)
    preflight = json.loads(preflight_p.read_text(encoding="utf-8"))
    cell_neffs = {
        tuple(int(x) for x in k.replace("G", "").split("_m")): float(v)
        for k, v in preflight["cell_effective_supports"].items()
    }
    neff_arr = np.array([cell_neffs[c] for c in GRID_CELLS])
    contrasts = _build_contrasts(pooled_dists)
    surf_params = _run_surface_bootstrap(GRID_CELLS, neff_arr, pooled_dists)

    cell_accs = {
        f"G{g}_m{m}": {
            **_pack_estimate(pooled_dists[(g, m)]),
            "draw_dispersion": draw_dispersions[(g, m)],
            "n_eff": cell_neffs[(g, m)],
        }
        for g, m in GRID_CELLS
    }
    return {
        "cell_accuracies": cell_accs,
        "contrasts": contrasts,
        "surface_parameters": surf_params,
    }
