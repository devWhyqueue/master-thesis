"""Bootstrap distributions, contrasts, and support surface parameter extraction."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from decodability import exp2_split_paths
from imbalance_benchmark.analysis.inference.context import BootstrapContext
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
from breadth.calibrate import scaled_test_probabilities
from breadth.analyze.secondary import (
    build_secondary_results,
    draw_secondary_distributions,
    pack_estimate,
)
from breadth.surface import fit_candidate_models

__all__ = [
    "load_draw_record",
    "collect_cell_distributions",
    "compute_contrasts_and_surface",
]

CellDists = dict[tuple[int, int], np.ndarray]
CellSecondaries = dict[tuple[int, int], dict[str, np.ndarray]]


def load_draw_record(
    config: dict[str, Any], split_index: int, g: int, m: int, draw_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load test labels, predictions, raw and scaled probabilities of one draw."""
    paths = split_paths(ensure_dirs(config), split_index)
    d_dir = draw_dir(paths, g, m, draw_index)
    rec = read_run_record(
        d_dir, splits=("test",), array_fields=("labels", "preds", "probabilities")
    )
    if rec is None or "test" not in rec.get("splits", {}):
        raise RuntimeError(f"Missing run record at {d_dir}")
    t_data = rec["splits"]["test"]
    probs = np.asarray(t_data["probabilities"], dtype=np.float64)
    return (
        np.asarray(t_data["labels"]),
        np.asarray(t_data["preds"]),
        probs,
        scaled_test_probabilities(d_dir, probs),
    )


def _accumulate_secondary(
    totals: dict[str, np.ndarray], dists: dict[str, np.ndarray], divisor: int
) -> None:
    """Add one draw's secondary distributions into the cell's running mean."""
    for key, dist in dists.items():
        totals[key] = totals.get(key, 0.0) + dist / divisor


def _collect_cell_replicates(
    contexts: dict[int, BootstrapContext],
    config: dict[str, Any],
    class_names: list[str],
) -> tuple[
    dict[tuple[int, int], list[list[np.ndarray]]],
    dict[tuple[int, int], list[float]],
    CellSecondaries,
]:
    """Sample bootstrap distributions for each split, cell, and draw."""
    n_classes = len(class_names)
    cell_splits: dict[tuple[int, int], list[list[Any]]] = {
        c: [[None] * N_DRAWS for _ in range(N_SPLITS)] for c in GRID_CELLS
    }
    raw_points: dict[tuple[int, int], list[float]] = {c: [] for c in GRID_CELLS}
    secondaries: CellSecondaries = {c: {} for c in GRID_CELLS}
    n_fits = N_SPLITS * N_DRAWS

    for s_idx in range(N_SPLITS):
        ctx = contexts[s_idx]
        for g, m in GRID_CELLS:
            for d_idx in range(N_DRAWS):
                arrays = load_draw_record(config, s_idx, g, m, d_idx)
                labels, preds = arrays[0], arrays[1]
                dist = (
                    ctx.ba_distribution(labels, preds[np.newaxis, :], n_classes) * 100.0
                )
                cell_splits[(g, m)][s_idx][d_idx] = dist
                raw_points[(g, m)].append(float(dist[0]))
                _accumulate_secondary(
                    secondaries[(g, m)],
                    draw_secondary_distributions(ctx, arrays, class_names),
                    n_fits,
                )
    return cell_splits, raw_points, secondaries


def collect_cell_distributions(
    config: dict[str, Any], class_names: list[str]
) -> tuple[CellDists, dict[tuple[int, int], float], CellSecondaries]:
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
    cell_splits, raw_points, secondaries = _collect_cell_replicates(
        contexts, config, class_names
    )
    pooled_dists: CellDists = {}
    dispersions: dict[tuple[int, int], float] = {}

    for c in GRID_CELLS:
        split_means = [np.mean(cell_splits[c][s], axis=0) for s in range(N_SPLITS)]
        pooled_dists[c] = np.mean(split_means, axis=0)
        dispersions[c] = float(np.std(raw_points[c]))
    return pooled_dists, dispersions, secondaries


def _fit_replicate_surface(
    cells: tuple[tuple[int, int], ...],
    neff_arr: np.ndarray,
    pooled_dists: CellDists,
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
    pooled_dists: CellDists,
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
    return {k: pack_estimate(np.array(vals)) for k, vals in reps.items()}


def _build_contrasts(pooled_dists: CellDists) -> dict[str, Any]:
    """Calculate and pack contrast bootstrap estimates."""
    delta_m = {g: pooled_dists[(g, 32)] - pooled_dists[(g, 8)] for g in BREADTH_LADDER}
    delta_g = {m: pooled_dists[(20, m)] - pooled_dists[(5, m)] for m in DEPTH_LADDER}
    x_dist = pooled_dists[(20, 8)] - pooled_dists[(5, 32)]
    return {
        "delta_m": {str(g): pack_estimate(delta_m[g]) for g in BREADTH_LADDER},
        "delta_g": {str(m): pack_estimate(delta_g[m]) for m in DEPTH_LADDER},
        "equal_budget_advantage_X": pack_estimate(x_dist),
        "isobudget_160_series": {
            "G20_m8": pack_estimate(pooled_dists[(20, 8)]),
            "G10_m16": pack_estimate(pooled_dists[(10, 16)]),
            "G5_m32": pack_estimate(pooled_dists[(5, 32)]),
        },
    }


def _cell_effective_supports(config: dict[str, Any]) -> dict[tuple[int, int], float]:
    """Read each cell's class-averaged effective support from signed preflight."""
    preflight_p = output_root(config) / "data" / "preflight.json"
    verify_signed_file(preflight_p)
    preflight = json.loads(preflight_p.read_text(encoding="utf-8"))
    supports: dict[tuple[int, int], float] = {}
    for key, value in preflight["cell_effective_supports"].items():
        g_text, m_text = key.replace("G", "").split("_m")
        supports[(int(g_text), int(m_text))] = float(value)
    return supports


def compute_contrasts_and_surface(
    config: dict[str, Any],
    pooled_dists: CellDists,
    draw_dispersions: dict[tuple[int, int], float],
    secondaries: CellSecondaries,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute contrasts and bootstrap distributions of support surface fits."""
    cell_neffs = _cell_effective_supports(config)
    neff_arr = np.array([cell_neffs[c] for c in GRID_CELLS])
    contrasts = _build_contrasts(pooled_dists)
    surf_params = _run_surface_bootstrap(GRID_CELLS, neff_arr, pooled_dists)

    cell_accs = {
        f"G{g}_m{m}": {
            **pack_estimate(pooled_dists[(g, m)]),
            "draw_dispersion": draw_dispersions[(g, m)],
            "n_eff": cell_neffs[(g, m)],
        }
        for g, m in GRID_CELLS
    }
    return {
        "cell_accuracies": cell_accs,
        "contrasts": contrasts,
        "surface_parameters": surf_params,
        "secondary": build_secondary_results(secondaries, GRID_CELLS, class_names),
    }
