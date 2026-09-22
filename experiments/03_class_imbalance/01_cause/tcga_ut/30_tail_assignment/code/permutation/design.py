"""Data assembly for the piecewise model: reading stored fits into (C, N, R) design arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from centre import N_DRAWS, N_SPLITS

from sites import allocation_dir

from assignment._io import paths_by_split, require_record
from assignment.properties import _spearman

from permutation import REUSED_ARMS
from permutation.model import leave_one_draw_out, z_of_counts

__all__ = [
    "N_FITS",
    "fit_index",
    "lodo",
    "model_design",
    "native_draws",
    "stored_counts",
]

N_FITS = N_SPLITS * N_DRAWS


def stored_counts(config: dict[str, Any], names: list[str], arm: str) -> np.ndarray:
    """(F, C) each stored fit's realized class_counts for one arm, canonical class order."""
    paths = paths_by_split(config)
    rows = [
        [
            require_record(allocation_dir(paths[s], arm, d))["class_counts"][c]
            for c in names
        ]
        for s in range(N_SPLITS)
        for d in range(N_DRAWS)
    ]
    return np.asarray(rows, dtype=np.float64)


def model_design(
    exp25_config: dict[str, Any],
    names: list[str],
    class_acc: dict[str, np.ndarray],
    w: np.ndarray,
    g: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack (arm, fit) observations into (C, N, R) z / delta / weight design arrays.

    z is fixed per (arm, fit) (repeated across the replicate axis); delta and weight already
    vary per replicate (own recall, and the draw-resampling weight, respectively).
    """
    _, n_classes, n_replicates = class_acc["r1"].shape  # class_acc arrays are (F, C, R)
    z = np.stack(
        [z_of_counts(stored_counts(exp25_config, names, a), g) for a in REUSED_ARMS]
    )
    delta = np.stack([class_acc[a] - class_acc["r1"] for a in REUSED_ARMS])
    n_arms = len(REUSED_ARMS)
    z_c = np.transpose(z, (2, 0, 1)).reshape(n_classes, n_arms * N_FITS)
    delta_c = np.transpose(delta, (2, 0, 1, 3)).reshape(
        n_classes, n_arms * N_FITS, n_replicates
    )
    z_c = np.broadcast_to(z_c[:, :, None], delta_c.shape)
    w_c = np.broadcast_to(np.tile(w, (n_arms, 1))[None], delta_c.shape)
    return z_c, delta_c, w_c


def fit_index() -> np.ndarray:
    """(A*F,) the (split, draw) fit index of each stacked (arm, fit) row."""
    return np.tile(np.arange(N_FITS), len(REUSED_ARMS))


def lodo(z_c: np.ndarray, delta_c: np.ndarray) -> dict[str, Any]:
    """Leave-one-draw-out prediction of each exp-25 draw's observed r100 D."""
    r100_col = REUSED_ARMS.index("r100") * N_FITS + np.arange(N_FITS)
    predicted, observed = leave_one_draw_out(
        z_c[:, :, 0], delta_c[:, :, 0], fit_index(), r100_col
    )
    return {
        "spearman": _spearman(list(predicted), list(observed)),
        "predicted": predicted.tolist(),
        "observed": observed.tolist(),
    }


def native_draws(delta_c: np.ndarray, rho: int) -> np.ndarray:
    """(F,) each exp-25 draw's own observed D at one ratio (point estimate)."""
    col = REUSED_ARMS.index(f"r{rho}") * N_FITS + np.arange(N_FITS)
    return -delta_c[:, col, 0].mean(axis=0)
