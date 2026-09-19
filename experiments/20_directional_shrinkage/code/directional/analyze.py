"""Analyze stage: pool reused R/C/S/St/RWc arms with new directional-shrinkage arms."""

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

from directional import ARMS, REUSED_ARM_FAMILIES, baseline_config

__all__ = ["derived", "run_analyze"]

_GAP_FAMILIES = ("R", "A", "At")
_RECOVERY_STEPS = (5, 10)
_COMPARE_FAMILIES = ("S", "St", "RWc")


def derived(arm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remedy gains against R, headroom recovery against C, gap shares, and remedy contrasts."""
    out: dict[str, np.ndarray] = {}
    for lo, hi in combinations(PATIENT_COUNTS, 2):
        step = f"{lo}_to_{hi}"
        for fam in _GAP_FAMILIES:
            out[f"gap_{fam}_{step}"] = arm[f"{fam}{hi}"] - arm[f"{fam}{lo}"]
        for fam in ("A", "At"):
            out[f"share_{fam}_{step}"] = (
                1.0 - out[f"gap_{fam}_{step}"] / out[f"gap_R_{step}"]
            )
    for g in PATIENT_COUNTS:
        for fam in ("A", "At", *_COMPARE_FAMILIES):
            out[f"{fam}{g}_minus_R{g}"] = arm[f"{fam}{g}"] - arm[f"R{g}"]
        for fam in _COMPARE_FAMILIES:
            out[f"At{g}_minus_{fam}{g}"] = arm[f"At{g}"] - arm[f"{fam}{g}"]
    for g in _RECOVERY_STEPS:
        for fam in ("A", "At"):
            out[f"recovery_{fam}_{g}"] = (arm[f"{fam}{g}"] - arm[f"R{g}"]) / (
                arm[f"C{g}"] - arm[f"R{g}"]
            )
    return out


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool reused and new arm accuracy over splits and draws and write estimates with 95% intervals."""
    names = canonical_class_names(config)
    acc: dict[str, np.ndarray] = {}
    for key, families in REUSED_ARM_FAMILIES.items():
        arms = tuple(f"{fam}{g}" for fam in families for g in PATIENT_COUNTS)
        acc.update(arm_accuracy(baseline_config(config, key), names, arms=arms))
    acc.update(arm_accuracy(config, names, arms=ARMS))
    n_replicates = next(iter(acc.values())).shape[-1]
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split, N_DRAWS, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    arm = {name: pooled(a, w) for name, a in acc.items()}
    dists = {f"arm_{name}": dist for name, dist in arm.items()} | derived(arm)
    return _write_analysis(config, acc, fit_split, dists)
