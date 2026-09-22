"""Analyze stage: pool reused R/C arms with new S/St shrinkage arms, gains, recovery, and shares."""

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

from shrinkage import ARMS, REUSED_ARMS, baseline_config

__all__ = ["derived", "run_analyze"]

_GAP_FAMILIES = ("R", "S")
_RECOVERY_STEPS = (5, 10)


def derived(arm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remedy gains, headroom recovery, and patient-count gap shares against the real cohort."""
    out: dict[str, np.ndarray] = {}
    for lo, hi in combinations(PATIENT_COUNTS, 2):
        step = f"{lo}_to_{hi}"
        for fam in _GAP_FAMILIES:
            out[f"gap_{fam}_{step}"] = arm[f"{fam}{hi}"] - arm[f"{fam}{lo}"]
        out[f"share_S_{step}"] = 1.0 - out[f"gap_S_{step}"] / out[f"gap_R_{step}"]
    for g in PATIENT_COUNTS:
        out[f"S{g}_minus_R{g}"] = arm[f"S{g}"] - arm[f"R{g}"]
        out[f"St{g}_minus_R{g}"] = arm[f"St{g}"] - arm[f"R{g}"]
    for g in _RECOVERY_STEPS:
        out[f"recovery_S_{g}"] = (arm[f"S{g}"] - arm[f"R{g}"]) / (
            arm[f"C{g}"] - arm[f"R{g}"]
        )
    return out


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool reused and new arm accuracy over splits and draws and write estimates with 95% intervals."""
    names = canonical_class_names(config)
    acc = {
        **arm_accuracy(baseline_config(config), names, arms=REUSED_ARMS),
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
