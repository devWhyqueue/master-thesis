"""Census stage: every random and composition cohort's quantities, guarded and signed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imbalance_benchmark.common import (
    output_root,
    sign_file,
    verify_signed_file,
    write_json,
)

from redundancy.analyze import _clipped_split_mean, _load_correlations

from neighbours.census import _load_manifests, load_allocations
from neighbours.embedding import assert_consistent_patch_counts, patient_class_means

from coverage_redundancy import N_SPLITS, exp6_config, exp10_config
from coverage_redundancy.cohorts import CensusContext, split_cohorts

__all__ = ["run_census", "load_quantities"]


def _load_manifests_and_means(
    config: dict[str, Any],
) -> tuple[
    list[str],
    dict[int, pd.DataFrame],
    dict[int, pd.DataFrame],
    dict[tuple[str, str], np.ndarray],
]:
    """Class names, full/train manifests, and split-invariant patient-class means."""
    class_names, full_dfs, train_dfs = _load_manifests(config)
    means, counts = patient_class_means(full_dfs[0])
    for s in range(1, N_SPLITS):
        assert_consistent_patch_counts(counts, full_dfs[s])
    return class_names, full_dfs, train_dfs, means


def _load_rho_bar(
    config: dict[str, Any], class_names: list[str]
) -> tuple[dict[str, float], np.ndarray]:
    """Class-mean full-feature ICC rho_bar_c, reused from exp-6 (Sec. "similarity")."""
    raw, correlations_class_names = _load_correlations(exp6_config(config))
    if correlations_class_names != class_names:
        raise RuntimeError(
            "exp-6 correlations class order does not match exp-2 freeze meta"
        )
    rho_by_class = dict(zip(class_names, _clipped_split_mean(raw["full"])[0]))
    return rho_by_class, raw["full"]


def run_census(config: dict[str, Any]) -> Path:
    """Census every random and composition cohort's tau^2, omega, Neff^omega, r.

    Computed before any recall loading so a failing reproduction of exp-6's
    ICC sample, or a coverage-distance mismatch against exp-10's own census,
    surfaces before the analyze stage ever runs.
    """
    class_names, full_dfs, train_dfs, means = _load_manifests_and_means(config)
    rho_by_class, raw_full = _load_rho_bar(config, class_names)
    composition_allocations = load_allocations(exp10_config(config))
    census = CensusContext(
        means, class_names, rho_by_class, raw_full, composition_allocations
    )

    cohorts: list[dict[str, Any]] = []
    for s in range(N_SPLITS):
        cohorts.extend(split_cohorts(config, s, full_dfs[s], train_dfs[s], census))

    out_p = output_root(config) / "data" / "quantities.json"
    write_json(out_p, {"cohorts": cohorts})
    sign_file(out_p)
    return out_p


def load_quantities(config: dict[str, Any]) -> dict[str, Any]:
    """Load and verify the signed per-cohort quantities census."""
    quantities_p = output_root(config) / "data" / "quantities.json"
    verify_signed_file(quantities_p)
    return json.loads(quantities_p.read_text(encoding="utf-8"))
