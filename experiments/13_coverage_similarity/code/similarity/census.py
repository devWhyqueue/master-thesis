"""Census stage: cohort search and the manipulation check (report Sec. "check")."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imbalance_benchmark.common import output_root, sign_file, write_json

from breadth.analyze.canonical import canonical_class_names
from breadth.analyze.diagnostics import _mean_leaves

from neighbours.census import _load_manifests
from neighbours.embedding import assert_consistent_patch_counts, patient_class_means

from coverage_redundancy.census import _load_rho_bar

from similarity import (
    MIN_OMEGA_GAP,
    MIN_R_GAP,
    N_SPLITS,
    NEFF_REL_TOL,
    OMEGA_TOL,
    R_TOL,
)
from similarity.descriptives import class_descriptives
from similarity.geometry import SplitInputs, build_split_geometry
from similarity.precision import load_precision
from similarity.search import CohortResult, search_split_draw

__all__ = ["run_census"]

_COVERAGE_PAIRS = (("good_low", "good_high"), ("poor_low", "poor_high"))
_SIMILARITY_PAIRS = (("good_low", "poor_low"), ("good_high", "poor_high"))
_GAP_KEYS = (
    "coverage_poor_minus_good_low",
    "coverage_poor_minus_good_high",
    "similarity_high_minus_low_good",
    "similarity_high_minus_low_poor",
)


def _pair_ok(
    results: dict[str, CohortResult], a: str, b: str, key: str, tol: float
) -> bool:
    return abs(getattr(results[a], key) - getattr(results[b], key)) <= tol


def _draw_checks(results: dict[str, CohortResult]) -> dict[str, bool]:
    """One (split, class, draw)'s matching-tolerance checks, on validation and training r."""
    checks: dict[str, bool] = {}
    for a, b in _COVERAGE_PAIRS:
        checks[f"{a}_vs_{b}_r_val"] = _pair_ok(results, a, b, "r_val", R_TOL)
        checks[f"{a}_vs_{b}_r_train"] = _pair_ok(results, a, b, "r_train", R_TOL)
    for a, b in _SIMILARITY_PAIRS:
        checks[f"{a}_vs_{b}_omega"] = _pair_ok(results, a, b, "omega", OMEGA_TOL)
    gl, tm = results["good_low"], results["ten_match"]
    checks["ten_match_r_val"] = abs(tm.r_val - gl.r_val) <= R_TOL
    checks["ten_match_r_train"] = abs(tm.r_train - gl.r_train) <= R_TOL
    checks["ten_match_neff_rel"] = (
        gl.neff is not None
        and tm.neff is not None
        and gl.neff != 0
        and abs(tm.neff / gl.neff - 1.0) <= NEFF_REL_TOL
    )
    return checks


def _manipulation_gaps(results: dict[str, CohortResult]) -> dict[str, float]:
    """Manipulated-gap descriptives (report Sec. "check"); a negative gap is a reversal."""
    return {
        "coverage_poor_minus_good_low": results["poor_low"].r_val
        - results["good_low"].r_val,
        "coverage_poor_minus_good_high": results["poor_high"].r_val
        - results["good_high"].r_val,
        "similarity_high_minus_low_good": results["good_high"].omega
        - results["good_low"].omega,
        "similarity_high_minus_low_poor": results["poor_high"].omega
        - results["poor_low"].omega,
    }


def _class_summary(draws: list[dict[str, CohortResult]], geo: Any) -> dict[str, Any]:
    """One class's checks, gaps, reversals, cohort descriptives, and site diversity."""
    checks_by_draw = [_draw_checks(r) for r in draws]
    gaps = [_manipulation_gaps(r) for r in draws]
    mean_gaps = _mean_leaves(gaps)
    return {
        "pass": all(all(c.values()) for c in checks_by_draw),
        "draws_failed": [
            i for i, c in enumerate(checks_by_draw) if not all(c.values())
        ],
        "gaps": mean_gaps,
        "reversals": {key: sum(1 for g in gaps if g[key] < 0) for key in mean_gaps},
        **class_descriptives(draws, geo),
    }


def _mean_gap(classes: dict[str, Any], key: str) -> float:
    return float(np.mean([c["gaps"][key] for c in classes.values()]))


def _manipulation_pass(classes: dict[str, Any]) -> dict[str, float]:
    """One split's manipulated gaps, averaged over classes (report Sec. "check")."""
    return {key: _mean_gap(classes, key) for key in _GAP_KEYS}


def _split_summary(
    config: dict[str, Any],
    split_idx: int,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    shared: SplitInputs,
    canonical_names: list[str],
    n_draws: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One split's per-class summaries and its allocation record (search results)."""
    geometry_by_class = build_split_geometry(
        config, split_idx, full_df, train_df, shared
    )
    draws_by_class: dict[str, list[dict[str, CohortResult]]] = {
        c: [] for c in canonical_names
    }
    for draw_idx in range(n_draws):
        by_class = search_split_draw(
            split_idx, draw_idx, canonical_names, geometry_by_class
        )
        for c_name, results in by_class.items():
            draws_by_class[c_name].append(results)

    classes = {
        c_name: _class_summary(draws_by_class[c_name], geometry_by_class[c_name])
        for c_name in canonical_names
    }
    gaps = _manipulation_pass(classes)
    manipulation_pass = (
        gaps["coverage_poor_minus_good_low"] >= MIN_R_GAP
        and gaps["coverage_poor_minus_good_high"] >= MIN_R_GAP
        and gaps["similarity_high_minus_low_good"] >= MIN_OMEGA_GAP
        and gaps["similarity_high_minus_low_poor"] >= MIN_OMEGA_GAP
    )
    matching_pass = all(c["pass"] for c in classes.values())
    summary = {
        "pass": matching_pass and manipulation_pass,
        "matching_pass": matching_pass,
        "manipulation_pass": manipulation_pass,
        "gaps": gaps,
        "classes": classes,
    }
    allocations = {
        c_name: {
            cohort_name: [
                draws_by_class[c_name][d][cohort_name].patients for d in range(n_draws)
            ]
            for cohort_name in draws_by_class[c_name][0]
        }
        for c_name in canonical_names
    }
    return summary, allocations


def _load_inputs(
    config: dict[str, Any],
) -> tuple[list[str], dict[int, pd.DataFrame], dict[int, pd.DataFrame], SplitInputs]:
    class_names, full_dfs, train_dfs = _load_manifests(config)
    means, counts = patient_class_means(full_dfs[0])
    for s in range(1, N_SPLITS):
        assert_consistent_patch_counts(counts, full_dfs[s])
    rho_by_class, raw_full = _load_rho_bar(config, class_names)
    shared = SplitInputs(means, class_names, rho_by_class, raw_full)
    return class_names, full_dfs, train_dfs, shared


def _write_outputs(
    config: dict[str, Any],
    n_draws: int,
    splits: dict[str, Any],
    allocations: dict[str, Any],
) -> Path:
    census_p = output_root(config) / "data" / "census.json"
    write_json(
        census_p,
        {
            "draws": n_draws,
            "r_tol": R_TOL,
            "omega_tol": OMEGA_TOL,
            "neff_rel_tol": NEFF_REL_TOL,
            "min_r_gap": MIN_R_GAP,
            "min_omega_gap": MIN_OMEGA_GAP,
            "splits": splits,
        },
    )
    sign_file(census_p)

    allocations_p = output_root(config) / "data" / "allocations.json"
    write_json(allocations_p, allocations)
    sign_file(allocations_p)
    return census_p


def _gate(splits: dict[str, Any]) -> None:
    failing = [s for s, v in splits.items() if not v["pass"]]
    if failing:
        raise RuntimeError(
            f"Manipulation check failed for split(s) {failing}; ending the experiment "
            "before fitting, with the original tolerances and class set retained."
        )


def run_census(config: dict[str, Any]) -> Path:
    """Search all cohorts and run the manipulation check; write, sign, and gate.

    Writes before gating so a failing census is still inspectable, and raises
    afterwards so the downstream ``afterok`` fit array never starts.
    """
    n_draws = load_precision(config)["selected_draws"]
    if not n_draws:
        raise RuntimeError(
            "precision.json has no selected_draws; cannot run the census"
        )

    canonical_names = canonical_class_names(config)
    _, full_dfs, train_dfs, shared = _load_inputs(config)

    splits: dict[str, Any] = {}
    allocations: dict[str, Any] = {}
    for s in range(N_SPLITS):
        splits[str(s)], allocations[str(s)] = _split_summary(
            config, s, full_dfs[s], train_dfs[s], shared, canonical_names, n_draws
        )

    census_p = _write_outputs(config, n_draws, splits, allocations)
    _gate(splits)
    return census_p
