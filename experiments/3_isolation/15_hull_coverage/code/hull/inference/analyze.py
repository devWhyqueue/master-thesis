"""Analyze stage: the class-recall model, the four parts, hull absorption, and the 10-to-20 prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths, write_json

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names, canonical_permutation
from breadth.analyze.secondary import pack_estimate

from sites.recall import contexts

from neighbours.accuracy import recall_stack

from decomposition.census import load_allocations
from decomposition.precision import load_precision

from hull import N_SPLITS
from hull.fit import fit_dir
from hull.inference import PATIENT_COUNTS, PREDICTORS
from hull.inference.model import (
    CellMeans,
    draw_weights,
    estimate,
    parts,
    prediction,
    random_means,
    reading,
)

__all__ = ["CensusArrays", "weighted_mean", "run_analyze"]

_NO_HULL = [0, 2, 3]  # r, omega, z


class CensusArrays(NamedTuple):
    """Achieved predictors (B, 2, C, 4), random mask (B, 2, C), cell labels (B, 2, C), and G=20 means."""

    x: np.ndarray
    random: np.ndarray
    labels: np.ndarray
    twenty: CellMeans


def _cell_label(row: dict[str, Any]) -> str:
    if row["mean_level"] is None:
        return "random"
    return f"mean{row['mean_level']:g}_hull{row['hull_level']:g}"


def _census_arrays(
    config: dict[str, Any], names: list[str], n_draws: int
) -> CensusArrays:
    shape = (N_SPLITS * n_draws, 2, len(names))
    x, random = np.full((*shape, len(PREDICTORS)), np.nan), np.zeros(shape, dtype=bool)
    labels = np.empty(shape, dtype=object)
    twenty: list[tuple[float, float, float]] = []
    for s in range(N_SPLITS):
        for row in load_allocations(config, s)["rows"]:
            values = (row["r_val"], row["h_val"], row["omega"])
            if row["g"] == 20:
                twenty.append(values)
                continue
            gi = PATIENT_COUNTS.index(row["g"])
            pos = (s * n_draws + row["draw"], gi, names.index(row["class"]))
            x[pos], random[pos], labels[pos] = (
                (*values, float(gi)),
                row["mean_level"] is None,
                _cell_label(row),
            )
    if np.isnan(x).any():
        raise RuntimeError(
            "Census rows do not cover every (split, draw, class, patient count)"
        )
    return CensusArrays(x, random, labels, CellMeans(*np.mean(twenty, axis=0)))


def _recall(
    config: dict[str, Any], names: list[str], g: int, n_draws: int
) -> np.ndarray:
    """(B, C, R) class recall in percent of the G-patient classifiers, one per (split, draw)."""
    ctxs = contexts(config)
    paths = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    perms = {s: canonical_permutation(config, s, names) for s in range(N_SPLITS)}
    keys = [(s, d) for s in range(N_SPLITS) for d in range(n_draws)]
    dirs = [fit_dir(paths[s], g, d) for s, d in keys]
    return (
        recall_stack(
            dirs, [ctxs[s] for s, _ in keys], [perms[s] for s, _ in keys], len(names)
        )
        * 100.0
    )


def weighted_mean(values: np.ndarray, mask: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(R,) mean of (B, C, R) values over masked (B, C) rows, with (B, R) block weights."""
    num = np.einsum("br,bcr->r", w, values * mask[..., None])
    return num / (mask.sum(axis=1) @ w)


def _gaps(
    y: np.ndarray, y20: np.ndarray, random: np.ndarray, w: np.ndarray
) -> dict[str, np.ndarray]:
    ran5 = weighted_mean(y[:, 0], random[:, 0], w)
    ran10 = weighted_mean(y[:, 1], random[:, 1], w)
    all20 = weighted_mean(y20, np.ones(y20.shape[:2], dtype=bool), w)
    return {"gap_5_to_10": ran10 - ran5, "gap_10_to_20": all20 - ran10}


def _cell_table(arrays: CensusArrays, y: np.ndarray) -> dict[str, Any]:
    """Descriptive mean recall and achieved validation r and h per (patient count, cell)."""
    table: dict[str, Any] = {}
    for gi, g in enumerate(PATIENT_COUNTS):
        for label in sorted(set(arrays.labels[:, gi].ravel())):
            mask = arrays.labels[:, gi] == label
            table[f"G{g}_{label}"] = {
                "recall": float(y[:, gi, :, 0][mask].mean()),
                "r_val": float(arrays.x[:, gi, :, 0][mask].mean()),
                "h_val": float(arrays.x[:, gi, :, 1][mask].mean()),
                "n": int(mask.sum()),
            }
    return table


def _distributions(
    config: dict[str, Any], names: list[str], n_draws: int
) -> tuple[dict[str, np.ndarray], CensusArrays, np.ndarray]:
    arrays = _census_arrays(config, names, n_draws)
    y = np.stack([_recall(config, names, g, n_draws) for g in PATIENT_COUNTS], axis=1)
    y20 = _recall(config, names, 20, n_draws)
    block_split = np.repeat(np.arange(N_SPLITS), n_draws)
    w = draw_weights(
        block_split, n_draws, y.shape[-1], np.random.default_rng(BOOTSTRAP_SEED)
    )
    beta = estimate(y, arrays.x, block_split, w)
    beta_no_hull = estimate(y, arrays.x[..., _NO_HULL], block_split, w)
    dists = {
        **parts(beta, random_means(arrays.x, arrays.random)),
        **_gaps(y, y20, arrays.random, w),
    }
    dists["absorption"] = beta_no_hull[:, 2] - beta[:, 3]
    dists["calibration"] = (
        dists["C"] + dists["H"] + dists["S"] + dists["P"] - dists["gap_5_to_10"]
    )
    dists["prediction_10_to_20"] = prediction(
        beta, random_means(arrays.x, arrays.random)[10], arrays.twenty
    )
    dists["prediction_error"] = dists["prediction_10_to_20"] - dists["gap_10_to_20"]
    dists.update({f"beta_{name}": beta[:, k] for k, name in enumerate(PREDICTORS)})
    return dists, arrays, y


def _within(est: dict[str, float]) -> bool:
    return est["ci_2_5"] >= -1.0 and est["ci_97_5"] <= 1.0


def _readings(packed: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Part readings and whether hull coverage operationalises the patient-count benefit."""
    readings = {
        p: reading((packed[p]["ci_2_5"], packed[p]["ci_97_5"]))
        for p in ("C", "H", "S", "P")
    }
    operationalised = (
        readings["H"] == "contributes"
        and readings["P"] in ("at_most_small", "counteracts")
        and _within(packed["calibration"])
        and _within(packed["prediction_error"])
    )
    return {"readings": readings, "operationalised": bool(operationalised)}


def run_analyze(config: dict[str, Any]) -> Path:
    """Fit the class-recall model and write parts, readings, prediction, and the cell table."""
    names = canonical_class_names(config)
    n_draws = load_precision(config)["selected_draws"]
    if not n_draws:
        raise RuntimeError("precision.json has no selected_draws; cannot analyze")
    dists, arrays, y = _distributions(config, names, n_draws)
    packed = {name: pack_estimate(dist) for name, dist in dists.items()}
    means = random_means(arrays.x, arrays.random)
    out = {
        "draws": n_draws,
        "estimates": packed,
        **_readings(packed),
        "random_cells": {str(g): m._asdict() for g, m in means.items()},
        "twenty": arrays.twenty._asdict(),
        "cells": _cell_table(arrays, y),
    }
    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, out)
    np.savez(
        output_root(config) / "data" / "distributions.npz",
        **cast(dict[str, Any], dists),
    )
    return out_p
