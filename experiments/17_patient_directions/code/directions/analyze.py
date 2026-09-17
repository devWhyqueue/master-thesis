"""Analyze stage: pool reused exp-16 arms with new whitened arms, gaps, shares, and interaction."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names
from breadth.analyze.secondary import pack_estimate

from decomposition.model import draw_weights

from centre import N_DRAWS, N_SPLITS, PATIENT_COUNTS
from centre.analyze import _per_split, arm_accuracy, pooled

from directions import ARMS, REUSED_ARMS, exp16_config

__all__ = ["derived", "run_analyze"]

_GAP_FAMILIES = ("R", "C", "CW", "RW", "RWc")


def derived(arm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gaps per family and step, whitening/centre shares, additivity, and gains to the real cohorts."""
    out: dict[str, np.ndarray] = {}
    for lo, hi in combinations(PATIENT_COUNTS, 2):
        step = f"{lo}_to_{hi}"
        for fam in _GAP_FAMILIES:
            out[f"gap_{fam}_{step}"] = arm[f"{fam}{hi}"] - arm[f"{fam}{lo}"]
        real = out[f"gap_R_{step}"]
        out[f"share_C_{step}"] = 1.0 - out[f"gap_C_{step}"] / real
        out[f"share_CW_{step}"] = 1.0 - out[f"gap_CW_{step}"] / real
        out[f"share_W_{step}"] = 1.0 - out[f"gap_RW_{step}"] / real
        out[f"share_Wc_{step}"] = 1.0 - out[f"gap_RWc_{step}"] / real
        out[f"share_interaction_{step}"] = (
            out[f"share_CW_{step}"] - out[f"share_C_{step}"] - out[f"share_W_{step}"]
        )
    for g in PATIENT_COUNTS:
        out[f"interaction_{g}"] = (arm[f"CW{g}"] - arm[f"C{g}"]) - (
            arm[f"RW{g}"] - arm[f"R{g}"]
        )
        out[f"RW{g}_minus_R{g}"] = arm[f"RW{g}"] - arm[f"R{g}"]
        out[f"RWc{g}_minus_R{g}"] = arm[f"RWc{g}"] - arm[f"R{g}"]
        out[f"CW{g}_minus_C{g}"] = arm[f"CW{g}"] - arm[f"C{g}"]
    return out


def _write_analysis(
    config: dict[str, Any],
    acc: dict[str, np.ndarray],
    fit_split: np.ndarray,
    dists: dict[str, np.ndarray],
) -> Path:
    """Write analysis.json (estimates and per-split accuracy) and the raw distributions."""
    out_p = output_root(config) / "data" / "analysis.json"
    write_json(
        out_p,
        {
            "draws_per_split": N_DRAWS,
            "estimates": {k: pack_estimate(v) for k, v in dists.items()},
            "per_split_arm_accuracy": _per_split(acc, fit_split),
        },
    )
    np.savez(
        output_root(config) / "data" / "distributions.npz",
        **cast(dict[str, Any], dists),
    )
    return out_p


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool reused and new arm accuracy over splits and draws and write estimates with 95% intervals."""
    names = canonical_class_names(config)
    acc = {
        **arm_accuracy(exp16_config(config), names, arms=REUSED_ARMS),
        **arm_accuracy(config, names, arms=ARMS),
    }
    n_replicates = next(iter(acc.values())).shape[-1]
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split, N_DRAWS, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    arm = {name: pooled(a, w) for name, a in acc.items()}
    dists = {f"arm_{name}": dist for name, dist in arm.items()} | derived(arm)
    return _write_analysis(config, acc, fit_split, dists)
