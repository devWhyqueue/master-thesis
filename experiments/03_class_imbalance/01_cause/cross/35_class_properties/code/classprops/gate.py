"""Manipulation check (0 new fits, r1 only): are BRACS's 7 classes inside the (headroom, margin)
range TCGA-UT's 30 classes already cover? If not, a class-composition explanation is untestable
with this design (BRACS would sit in property territory TCGA-UT's own classes never occupy), and
the experiment stops before spending any bootstrap/regression effort on the full pools.

Needs only each dataset's own stored r1 fits (``slurm.prevalence_outputs``) plus a minimal peer
descriptor (``slurm.gate_peer``: the other dataset's name, ``prevalence_outputs``, and
``exp2_outputs``), so either dataset's own config can run the gate standalone, in either order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root, write_json
from scipy.spatial import Delaunay

from breadth.analyze.canonical import canonical_class_names

from centre import N_SPLITS

from classprops import GATE_MIN_FRACTION
from classprops.covariates import (
    cross_fitted_headroom,
    cross_fitted_margin,
    shared_draw_weight,
)
from classprops.pool import FIT_SPLIT, read_r1_stack

__all__ = ["run_gate", "write_gate"]


def _point(arr: np.ndarray) -> np.ndarray:
    """Replicate-0 (observed) slice, dropping the replicate axis."""
    return arr[..., 0]


def _peer_config(config: dict[str, Any]) -> dict[str, Any]:
    """A minimal standalone config for the peer dataset, from ``slurm.gate_peer``."""
    peer = config["slurm"]["gate_peer"]
    return {
        "dataset": {"name": peer["name"]},
        "slurm": {
            "prevalence_outputs": peer["prevalence_outputs"],
            "exp2_outputs": peer["exp2_outputs"],
        },
    }


def _dataset_points(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(S, C) cross-fitted headroom and margin point estimates for one dataset."""
    names = canonical_class_names(config)
    r1 = read_r1_stack(config, names)
    w = shared_draw_weight(FIT_SPLIT, r1.shape[-1], seed=0)
    h = _point(cross_fitted_headroom(r1, w, names))
    m = _point(cross_fitted_margin(config, names, r1.shape[-1]))
    return h, m


def _fraction_inside(hull_points: np.ndarray, query_points: np.ndarray) -> float:
    """Fraction of ``query_points`` inside the convex hull of ``hull_points``."""
    inside = Delaunay(hull_points).find_simplex(query_points) >= 0
    return float(np.mean(inside))


def run_gate(config: dict[str, Any]) -> dict[str, Any]:
    """Cross-fitted (headroom, margin) overlap between BRACS and TCGA-UT, from either config."""
    own_h, own_m = _dataset_points(config)
    peer_h, peer_m = _dataset_points(_peer_config(config))
    if config["dataset"]["name"] == "bracs":
        bracs_h, bracs_m, tcga_h, tcga_m = own_h, own_m, peer_h, peer_m
    else:
        bracs_h, bracs_m, tcga_h, tcga_m = peer_h, peer_m, own_h, own_m

    fractions = [
        _fraction_inside(
            np.stack([tcga_h[s], tcga_m[s]], axis=-1),
            np.stack([bracs_h[s], bracs_m[s]], axis=-1),
        )
        for s in range(N_SPLITS)
    ]
    fraction = float(np.mean(fractions))
    gate_pass = fraction >= GATE_MIN_FRACTION
    return {
        "fraction_inside_by_split": fractions,
        "fraction_inside": fraction,
        "threshold": GATE_MIN_FRACTION,
        "pass": gate_pass,
        "label": None if gate_pass else "dataset_specific_by_support",
    }


def write_gate(config: dict[str, Any], result: dict[str, Any]) -> None:
    """Write the gate result to this dataset's own output tree."""
    write_json(output_root(config) / "data" / "gate.json", result)
