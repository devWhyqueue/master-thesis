"""Analyze stage: pool reused and new span arms with contrasts and diagnostics."""

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

from directions import KAPPA_FACTORS

from span import (
    ARMS,
    CAPTURE_MULTIPLES,
    FIT_FAMILIES,
    REUSED_ARM_FAMILIES,
    baseline_config,
)

__all__ = ["derived", "run_analyze"]

_NEW_FAMILIES = (*FIT_FAMILIES, "Wt", "Rt")
_RECOVERY_STEPS = (5, 10)


def derived(arm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gains over R and RWc, specificity, headroom share, recovery, and the 5 -> 20 gap shares."""
    out: dict[str, np.ndarray] = {}
    for g in PATIENT_COUNTS:
        for fam in ("RWc", "RW", *_NEW_FAMILIES):
            out[f"{fam}{g}_minus_R{g}"] = arm[f"{fam}{g}"] - arm[f"R{g}"]
        for fam in _NEW_FAMILIES:
            out[f"{fam}{g}_minus_RWc{g}"] = arm[f"{fam}{g}"] - arm[f"RWc{g}"]
        out[f"RW{g}_minus_RWc{g}"] = arm[f"RW{g}"] - arm[f"RWc{g}"]
        out[f"Wt{g}_minus_Rt{g}"] = arm[f"Wt{g}"] - arm[f"Rt{g}"]
        for fam in ("Wt", "Rt"):
            out[f"headroom_share_{fam}{g}"] = (
                out[f"{fam}{g}_minus_RWc{g}"] / out[f"RW{g}_minus_RWc{g}"]
            )
    for g in _RECOVERY_STEPS:
        for fam in ("RWc", "Wt", "Rt"):
            out[f"recovery_{fam}_{g}"] = (arm[f"{fam}{g}"] - arm[f"R{g}"]) / (
                arm[f"C{g}"] - arm[f"R{g}"]
            )
    gap_r = arm["R20"] - arm["R5"]
    for fam in ("RWc", "Wt", "Rt"):
        out[f"share_{fam}_5_to_20"] = 1.0 - (arm[f"{fam}20"] - arm[f"{fam}5"]) / gap_r
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


def _kappa_factor(record: dict[str, Any]) -> float:
    """Unscaled kappa factor: the anchored factor's position in the score dict indexes ``KAPPA_FACTORS``."""
    scores = list(record["kappa_validation_scores"])
    return KAPPA_FACTORS[scores.index(str(record["kappa_factor"]))]


def _diagnostics(config: dict[str, Any], acc: dict[str, np.ndarray]) -> dict[str, Any]:
    """Selected s and kappa counts, mean captured pool share, and captured-vs-accuracy correlation."""
    diag: dict[str, Any] = {}
    for g in PATIENT_COUNTS:
        delta = (acc[f"Wt{g}"] - acc[f"RWc{g}"]).mean(axis=1)
        for kind in ("W", "R"):
            tuned = _records(config, f"{kind}t{g}")
            diag[f"s_counts_{kind}t{g}"] = dict(Counter(str(r["s"]) for r in tuned))
            diag[f"kappa_counts_{kind}t{g}"] = dict(
                Counter(str(_kappa_factor(r)) for r in tuned)
            )
            for m in CAPTURE_MULTIPLES:
                diag[f"mean_captured_{kind}t{g}_{m}"] = float(
                    np.mean([r["captured_top_k"][str(m)] for r in tuned])
                )
        diag[f"mean_added_pool_overlap_{g}"] = float(
            np.mean([r["added_pool_overlap"] for r in _records(config, f"Wc{g}")])
        )
        wt = _records(config, f"Wt{g}")
        for m in CAPTURE_MULTIPLES:
            gain = np.array(
                [r["captured_top_k"][str(m)] - r["captured_cohort"] for r in wt]
            )
            diag[f"corr_captured_accuracy_{g}_{m}"] = (
                float(np.corrcoef(gain, delta)[0, 1]) if gain.std() > 0 else None
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
    write_json(path.with_name("diagnostics.json"), _diagnostics(config, acc))
    return path
