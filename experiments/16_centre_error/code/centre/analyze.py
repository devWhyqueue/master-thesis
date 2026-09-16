"""Analyze stage: arm accuracy, patient-count gaps, centre shares, and centre-error contrasts."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths, write_json

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names, canonical_permutation
from breadth.analyze.secondary import pack_estimate

from sites import allocation_dir
from sites.recall import contexts

from neighbours.accuracy import recall_stack

from decomposition.model import draw_weights

from centre import ARMS, N_DRAWS, N_SPLITS, PATIENT_COUNTS

__all__ = ["arm_accuracy", "pooled", "derived", "run_analyze"]

_GAP_FAMILIES = ("R", "C", "N", "CW")


def arm_accuracy(config: dict[str, Any], names: list[str]) -> dict[str, np.ndarray]:
    """Per arm, (F, R) class-mean patient-macro recall in percent, one row per (split, draw) fit."""
    ctxs = contexts(config)
    keys = [(s, d) for s in range(N_SPLITS) for d in range(N_DRAWS)]
    paths = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    perms = {s: canonical_permutation(config, s, names) for s in range(N_SPLITS)}
    return {
        arm: recall_stack(
            [allocation_dir(paths[s], arm, d) for s, d in keys],
            [ctxs[s] for s, _ in keys],
            [perms[s] for s, _ in keys],
            len(names),
        ).mean(axis=1)
        * 100.0
        for arm in ARMS
    }


def pooled(acc: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(R,) draw-weighted mean over fits of (F, R) accuracy with (F, R) weights."""
    return (w * acc).sum(axis=0) / w.sum(axis=0)


def derived(arm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gaps per family and step, centre and combined shares, noise ratios, and contrasts to the real cohorts."""
    out: dict[str, np.ndarray] = {}
    for lo, hi in combinations(PATIENT_COUNTS, 2):
        step = f"{lo}_to_{hi}"
        for fam in _GAP_FAMILIES:
            out[f"gap_{fam}_{step}"] = arm[f"{fam}{hi}"] - arm[f"{fam}{lo}"]
        real = out[f"gap_R_{step}"]
        out[f"centre_share_{step}"] = 1.0 - out[f"gap_C_{step}"] / real
        out[f"combined_share_{step}"] = 1.0 - out[f"gap_CW_{step}"] / real
        out[f"noise_ratio_{step}"] = out[f"gap_N_{step}"] / real
    for g in PATIENT_COUNTS:
        out[f"N{g}_minus_R{g}"] = arm[f"N{g}"] - arm[f"R{g}"]
        out[f"C{g}_minus_R{g}"] = arm[f"C{g}"] - arm[f"R{g}"]
        out[f"CW{g}_minus_C{g}"] = arm[f"CW{g}"] - arm[f"C{g}"]
    out["C5_minus_R20"] = arm["C5"] - arm["R20"]
    for name in ("Swap5", "Glob5", "Disc5", "Off5"):
        out[f"{name}_minus_R5"] = arm[name] - arm["R5"]
    return out


def _per_split(acc: dict[str, np.ndarray], fit_split: np.ndarray) -> dict[str, Any]:
    """Observed mean accuracy per split and arm."""
    return {
        str(s): {name: float(a[fit_split == s, 0].mean()) for name, a in acc.items()}
        for s in range(N_SPLITS)
    }


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool every arm over splits and draws and write estimates with 95 % intervals."""
    names = canonical_class_names(config)
    acc = arm_accuracy(config, names)
    n_replicates = next(iter(acc.values())).shape[-1]
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split, N_DRAWS, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    arm = {name: pooled(a, w) for name, a in acc.items()}
    dists = {f"arm_{name}": dist for name, dist in arm.items()} | derived(arm)
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
