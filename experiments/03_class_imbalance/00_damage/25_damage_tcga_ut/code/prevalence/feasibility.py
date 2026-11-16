"""Feasibility check of the configured (P, n) design on the real train manifests (0 fits).

For every split and class it counts the patients holding at least ``depth`` patches
(``prevalence.depth_of``) and asserts, over the minimum across splits, that (a) every class has at
least P eligible patients, (b) the r100 head fits under the cap P x depth, and (c) the r100 tail keeps
at least one patch per patient. Prints the table behind issue #7's design choice.
"""

from __future__ import annotations

import logging

import json
from typing import Any

import pandas as pd
from breadth import N_SPLITS, exp2_split_paths
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.construction import allocate_counts

from prevalence import RATIOS, balanced_of, depth_of, patients_per_class

logger = logging.getLogger(__name__)

__all__ = ["check_feasibility"]

_COLUMNS = {"split", "cancer_type", "case_id"}


def _patch_counts(config: dict[str, Any]) -> dict[str, list[pd.Series]]:
    """Per class, each split's patches-per-patient counts of the train manifest (read once)."""
    out: dict[str, list[pd.Series]] = {}
    for split in range(N_SPLITS):
        paths = exp2_split_paths(config, split)
        manifest = pd.read_csv(
            paths["data"] / "manifest.csv", usecols=lambda c: c in _COLUMNS
        ).query("split == 'train'")
        for name in load_freeze_meta(paths)["class_names"]:
            cases = manifest.loc[manifest["cancer_type"] == name, "case_id"]
            out.setdefault(name, []).append(cases.astype(str).value_counts())
    return out


def _eligible_counts(
    counts: dict[str, list[pd.Series]], depth: int
) -> dict[str, list[int]]:
    """Per class, the number of patients holding at least ``depth`` patches in each split."""
    return {c: [int((s >= depth).sum()) for s in v] for c, v in counts.items()}


def _problems(g: int, n: int, depth: int, eligible: dict[str, list[int]]) -> list[str]:
    """Head-cap, tail-floor, budget and eligible-patient violations of the design."""
    k = len(eligible)
    total = g * n * k
    out = []
    for ratio in RATIOS:
        ranked = allocate_counts([g * depth] * k, total, float(ratio), g)
        if ranked[0] > g * depth or ranked[-1] < g or sum(ranked) != total:
            out.append(f"r{ratio}: head {ranked[0]}, tail {ranked[-1]}")
    short = {c: min(v) for c, v in eligible.items() if min(v) < g}
    if short:
        out.append(f"classes with < {g} eligible patients: {short}")
    return out


def check_feasibility(config: dict[str, Any]) -> dict[str, Any]:
    """Print and return the feasibility table; raise ``ValueError`` if the design is infeasible."""
    g = patients_per_class(config)
    n = balanced_of(g)
    counts = _patch_counts(config)
    k = len(counts)
    depth = depth_of(g, k)
    eligible = _eligible_counts(counts, depth)
    problems = _problems(int(g), n, depth, eligible)
    table = {
        "patients_per_class": int(g),
        "patches_per_patient": n,
        "num_classes": k,
        "eligibility_depth": depth,
        "budget": int(g) * n * k,
        "eligible_patients_min_over_splits": {c: min(v) for c, v in eligible.items()},
        "problems": problems,
    }
    logger.info(json.dumps(table, indent=2))
    if problems:
        raise ValueError("; ".join(problems))
    return table
