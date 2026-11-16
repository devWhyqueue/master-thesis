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
from breadth.sampling import eligible_patients_by_class
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.construction import allocate_counts

from prevalence import RATIOS, balanced_of, depth_of, patients_per_class

logger = logging.getLogger(__name__)

__all__ = ["check_feasibility"]


def _eligible_counts(config: dict[str, Any], depth: int) -> dict[str, list[int]]:
    """Per class, the number of eligible patients in each split's train manifest."""
    counts: dict[str, list[int]] = {}
    for split in range(N_SPLITS):
        paths = exp2_split_paths(config, split)
        manifest = pd.read_csv(paths["data"] / "manifest.csv").query("split == 'train'")
        for name in load_freeze_meta(paths)["class_names"]:
            counts.setdefault(name, []).append(
                len(eligible_patients_by_class(manifest, name, depth))
            )
    return counts


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
    k = len(_eligible_counts(config, 0))
    depth = depth_of(g, k)
    eligible = _eligible_counts(config, depth)
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
