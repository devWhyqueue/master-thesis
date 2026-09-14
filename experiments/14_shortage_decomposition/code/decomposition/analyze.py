"""Analyze stage: the class-recall model, the three parts, and their calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths, write_json

from breadth import BOOTSTRAP_SEED, N_SPLITS
from breadth.analyze.canonical import canonical_class_names, canonical_permutation
from breadth.analyze.secondary import pack_estimate

from sites import allocation_dir
from sites.recall import contexts

from neighbours.accuracy import recall_stack

from coverage_redundancy.census import load_quantities

from decomposition import exp12_config
from decomposition.census import load_allocations
from decomposition.model import draw_weights, estimate, parts, reading
from decomposition.precision import load_precision

__all__ = ["run_analyze"]


def _ctx_list(config: dict[str, Any], n_draws: int) -> list[Any]:
    ctxs = contexts(config)
    return [ctxs[s] for s in range(N_SPLITS) for _ in range(n_draws)]


def _perm_list(
    config: dict[str, Any], canonical_names: list[str], n_draws: int
) -> list[Any]:
    per_split = [
        canonical_permutation(config, s, canonical_names) for s in range(N_SPLITS)
    ]
    return [per_split[s] for s in range(N_SPLITS) for _ in range(n_draws)]


def _census_arrays(
    config: dict[str, Any], canonical_names: list[str], n_draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Achieved [r_val, omega, z] predictors and the two random-cell (fit, class) masks."""
    n_classes = len(canonical_names)
    n_fits = N_SPLITS * n_draws
    x = np.empty((n_fits, n_classes, 3), dtype=np.float64)
    mask5 = np.zeros((n_fits, n_classes), dtype=bool)
    mask10 = np.zeros_like(mask5)
    for s in range(N_SPLITS):
        rows = load_allocations(config, s)["rows"]
        by_draw_class = {(r["draw"], r["class"]): r for r in rows}
        for d in range(n_draws):
            f = s * n_draws + d
            for c_idx, c_name in enumerate(canonical_names):
                row = by_draw_class[(d, c_name)]
                z = 1.0 if row["g"] == 10 else 0.0
                x[f, c_idx] = (row["r_val"], row["omega"], z)
                if row["r_level"] is None:
                    (mask5 if row["g"] == 5 else mask10)[f, c_idx] = True
    return x, mask5, mask10


def _recall(
    config: dict[str, Any], canonical_names: list[str], n_draws: int
) -> np.ndarray:
    """The class-recall stack (F, C, R) of this experiment's own stored fits, in percent."""
    n_classes = len(canonical_names)
    ctx_l = _ctx_list(config, n_draws)
    perms = _perm_list(config, canonical_names, n_draws)
    paths14 = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    dirs = [
        allocation_dir(paths14[s], "cells", d)
        for s in range(N_SPLITS)
        for d in range(n_draws)
    ]
    return recall_stack(dirs, ctx_l, perms, n_classes) * 100.0


def _pooled_gap(y: np.ndarray, mask5: np.ndarray, mask10: np.ndarray) -> np.ndarray:
    """Pooled random-cell recall gap (R,): mean over masked (fit, class) rows."""
    f5, c5 = np.nonzero(mask5)
    f10, c10 = np.nonzero(mask10)
    return y[f10, c10, :].mean(axis=0) - y[f5, c5, :].mean(axis=0)


def _model_fit(
    config: dict[str, Any], canonical_names: list[str], n_draws: int
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    """Fit beta, compute the three parts, the observed random-cell gap, and their means."""
    x, mask5, mask10 = _census_arrays(config, canonical_names, n_draws)
    y = _recall(config, canonical_names, n_draws)
    fit_split = np.repeat(np.arange(N_SPLITS), n_draws)
    means = {
        "r_bar_ran5": float(x[:, :, 0][mask5].mean()),
        "r_bar_ran10": float(x[:, :, 0][mask10].mean()),
        "omega_bar_ran5": float(x[:, :, 1][mask5].mean()),
        "omega_bar_ran10": float(x[:, :, 1][mask10].mean()),
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    w = draw_weights(fit_split, n_draws, y.shape[-1], rng)
    beta = estimate(y, x, fit_split, w)
    pts = parts(beta, *means.values())
    delta_dist = _pooled_gap(y, mask5, mask10)
    return beta, pts, delta_dist, means


def _calibration(pts: dict[str, np.ndarray], delta_dist: np.ndarray) -> dict[str, Any]:
    """Calibration check: C+S+P against the observed random-cell gap (report Sec. "decomposition")."""
    calib_dist = pts["C"] + pts["S"] + pts["P"] - delta_dist
    calib_est = pack_estimate(calib_dist)
    calib_settled = not (calib_est["ci_2_5"] > 1.0 or calib_est["ci_97_5"] < -1.0)
    return {"calibration": calib_est, "settled": bool(calib_settled)}


def _pack_results(
    pts: dict[str, np.ndarray], delta_dist: np.ndarray
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    packed = {name: pack_estimate(dist) for name, dist in pts.items()}
    readings = {
        name: reading((est["ci_2_5"], est["ci_97_5"])) for name, est in packed.items()
    }
    return packed, readings, _calibration(pts, delta_dist)


def _secondary(config: dict[str, Any], beta: np.ndarray) -> dict[str, Any]:
    """Apply delta/lambda to exp-12's stored 5-patient cohorts; compare with the observed gain."""
    quantities = load_quantities(exp12_config(config))["cohorts"]
    shortage = [c for c in quantities if c["source"] == "shortage"]
    r_ran = float(np.mean([c["r"] for c in shortage if c["allocation"] == "random"]))
    r_dis = float(np.mean([c["r"] for c in shortage if c["allocation"] == "dispersed"]))
    omega_ran = float(
        np.mean([c["omega_mean"] for c in shortage if c["allocation"] == "random"])
    )
    omega_dis = float(
        np.mean([c["omega_mean"] for c in shortage if c["allocation"] == "dispersed"])
    )
    delta, lam = beta[:, 0], beta[:, 1]
    predicted = delta * (r_dis - r_ran) + lam * (omega_dis - omega_ran)

    npz_p = output_root(exp12_config(config)) / "data" / "distributions.npz"
    observed = np.load(npz_p)["all_delta_sel"]
    n = min(len(predicted), len(observed))
    error = predicted[:n] - observed[:n]
    return {
        "predicted": pack_estimate(predicted),
        "observed": pack_estimate(observed),
        "error": pack_estimate(error),
    }


def _build_output(
    n_draws: int,
    beta: np.ndarray,
    packed: dict[str, Any],
    readings: dict[str, str],
    delta_dist: np.ndarray,
    calibration: dict[str, Any],
    settled: bool,
    means: dict[str, float],
    secondary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the analysis.json payload."""
    return {
        "draws": n_draws,
        "beta": {
            "delta": pack_estimate(beta[:, 0]),
            "lambda": pack_estimate(beta[:, 1]),
            "gamma": pack_estimate(beta[:, 2]),
        },
        "parts": packed,
        "readings": readings,
        "delta_5_to_10": pack_estimate(delta_dist),
        "calibration": calibration,
        "settled": bool(settled),
        "random_cells": means,
        "secondary": secondary,
    }


def _write_outputs(
    config: dict[str, Any],
    out: dict[str, Any],
    pts: dict[str, np.ndarray],
    delta_dist: np.ndarray,
) -> Path:
    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, out)
    np.savez(
        output_root(config) / "data" / "distributions.npz",
        C=pts["C"],
        S=pts["S"],
        P=pts["P"],
        delta_5_to_10=delta_dist,
    )
    return out_p


def run_analyze(config: dict[str, Any]) -> Path:
    """Fit the class-recall model, compute the three parts, and their calibration."""
    canonical_names = canonical_class_names(config)
    n_draws = load_precision(config)["selected_draws"]
    if not n_draws:
        raise RuntimeError("precision.json has no selected_draws; cannot analyze")

    beta, pts, delta_dist, means = _model_fit(config, canonical_names, n_draws)
    packed, readings, calibration = _pack_results(pts, delta_dist)
    settled = calibration["settled"] and all(
        r != "unresolved" for r in readings.values()
    )
    secondary = _secondary(config, beta)

    out = _build_output(
        n_draws,
        beta,
        packed,
        readings,
        delta_dist,
        calibration,
        settled,
        means,
        secondary,
    )
    return _write_outputs(config, out, pts, delta_dist)
