"""Analyze stage: pool every arm's accuracy and write gaps, shares, noise ratio, and interaction."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names

from decomposition.model import draw_weights

from centre import N_DRAWS, N_SPLITS, PATIENT_COUNTS
from centre.analyze import arm_accuracy, pooled

from directions.analyze import _write_analysis
from directions.analyze import derived as _directions_derived

from mechanism import ARMS

__all__ = ["derived", "run_analyze"]


def derived(arm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Directions' gaps/shares/interaction plus the noise-family gap, ratio, and contrasts."""
    out = dict(_directions_derived(arm))
    for lo, hi in combinations(PATIENT_COUNTS, 2):
        step = f"{lo}_to_{hi}"
        out[f"gap_N_{step}"] = arm[f"N{hi}"] - arm[f"N{lo}"]
        out[f"noise_ratio_{step}"] = out[f"gap_N_{step}"] / out[f"gap_R_{step}"]
    for g in PATIENT_COUNTS:
        out[f"N{g}_minus_R{g}"] = arm[f"N{g}"] - arm[f"R{g}"]
        out[f"C{g}_minus_R{g}"] = arm[f"C{g}"] - arm[f"R{g}"]
    out["C5_minus_R20"] = arm["C5"] - arm["R20"]
    return out


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool every arm's accuracy over splits and draws and write estimates with 95% intervals."""
    names = canonical_class_names(config)
    acc = arm_accuracy(config, names, arms=ARMS)
    n_replicates = next(iter(acc.values())).shape[-1]
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split, N_DRAWS, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    arm = {name: pooled(a, w) for name, a in acc.items()}
    dists = {f"arm_{name}": dist for name, dist in arm.items()} | derived(arm)
    return _write_analysis(config, acc, fit_split, dists)
