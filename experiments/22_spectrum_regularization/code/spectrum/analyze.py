"""Analyze stage: pool reused R/C/RW/RWc arms with new spectrum arms and their diagnostics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import (
    ensure_dirs,
    read_run_record,
    split_paths,
    write_json,
)

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names

from decomposition.model import draw_weights

from sites import allocation_dir

from centre import N_DRAWS, N_SPLITS, PATIENT_COUNTS
from centre.analyze import arm_accuracy, pooled

from directions.analyze import _write_analysis

from spectrum import ARMS, BETAS, REUSED_ARM_FAMILIES, baseline_config

__all__ = ["derived", "run_analyze"]

_NEW_FAMILIES = ("F", "Ba", "Bb", "Bc", "Bt", "O", "P")
_RECOVERY_STEPS = (5, 10)


def derived(arm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gains over R and RWc, the RWc -> O -> RW decomposition, control, recovery, and Bt gap share."""
    out: dict[str, np.ndarray] = {}
    for g in PATIENT_COUNTS:
        for fam in ("RWc", "RW", *_NEW_FAMILIES):
            out[f"{fam}{g}_minus_R{g}"] = arm[f"{fam}{g}"] - arm[f"R{g}"]
        for fam in _NEW_FAMILIES:
            out[f"{fam}{g}_minus_RWc{g}"] = arm[f"{fam}{g}"] - arm[f"RWc{g}"]
        out[f"RW{g}_minus_O{g}"] = arm[f"RW{g}"] - arm[f"O{g}"]
        out[f"RWc{g}_minus_P{g}"] = arm[f"RWc{g}"] - arm[f"P{g}"]
    for g in _RECOVERY_STEPS:
        for fam in ("RWc", "Bt"):
            out[f"recovery_{fam}_{g}"] = (arm[f"{fam}{g}"] - arm[f"R{g}"]) / (
                arm[f"C{g}"] - arm[f"R{g}"]
            )
    gap_r = arm["R20"] - arm["R5"]
    out["share_Bt_5_to_20"] = 1.0 - (arm["Bt20"] - arm["Bt5"]) / gap_r
    return out


def _records(config: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    """Run records of one arm over all (split, draw) fits, without arrays."""
    paths = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    records = [
        read_run_record(allocation_dir(paths[s], arm, d), array_fields=())
        for s in range(N_SPLITS)
        for d in range(N_DRAWS)
    ]
    return [r for r in records if r is not None]


def _diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    """Selection counts of beta and kappa and mean O/P diagnostics per patient count."""
    diag: dict[str, Any] = {}
    for g in PATIENT_COUNTS:
        bt = _records(config, f"Bt{g}")
        diag[f"beta_counts_{g}"] = dict(Counter(str(r["beta"]) for r in bt))
        for fam in (*BETAS, "O", "P"):
            recs = _records(config, f"{fam}{g}")
            diag[f"kappa_counts_{fam}{g}"] = dict(
                Counter(str(r["kappa_factor"]) for r in recs)
            )
        for fam, key in (
            ("O", "oracle_slope"),
            ("O", "pool_variance_captured"),
            ("P", "subspace_overlap"),
        ):
            diag[f"mean_{key}_{g}"] = float(
                np.mean([r[key] for r in _records(config, f"{fam}{g}")])
            )
    return diag


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
    path = _write_analysis(config, acc, fit_split, dists)
    write_json(path.with_name("diagnostics.json"), _diagnostics(config))
    return path
