"""Census stage: exp-10 allocation prefixes, similarity quantities, and the rho_sel gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from imbalance_benchmark.common import output_root, sign_file, write_json

from breadth import N_SPLITS
from breadth.analyze.diagnostics import _mean_leaves

from neighbours.census import _load_manifests, load_allocations, load_census
from neighbours.embedding import (
    assert_consistent_patch_counts,
    embed,
    patient_class_means,
    training_mu,
)

from coverage_redundancy.census import _load_rho_bar
from coverage_redundancy.cohorts import _build_split_context
from coverage_redundancy.quantities import class_quantities, cohort_record

from shortage import ALLOCATIONS, MIN_RHO_SEL, N_DRAWS, exp10_config
from shortage.allocation import class_draw

__all__ = ["run_census"]

_R_TOLERANCE = 1e-9


class _SplitInputs(NamedTuple):
    """Cross-split state that every split's census stage shares."""

    means: dict[tuple[str, str], np.ndarray]
    class_names: list[str]
    rho_by_class: dict[str, float]
    raw_full: np.ndarray


def _class_rows(
    full_df: pd.DataFrame,
    embeddings: dict[tuple[str, str], np.ndarray],
    class_names: list[str],
    split_allocations: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Every class's five draws for one split, from exp-10's own allocations."""
    class_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in class_names}
    for c_name in class_names:
        for draw_idx in range(N_DRAWS):
            record = split_allocations["allocations"][str(draw_idx)][c_name]
            class_rows[c_name].append(class_draw(embeddings, full_df, c_name, record))
    return class_rows


def _ratios(r: dict[str, float]) -> dict[str, float]:
    """Selection ratio rho_sel (Eq. ratio)."""
    denom = r["r_5"] - r["r_ran20"]
    return {"rho_sel": (r["r_5"] - r["r_dispersed"]) / denom if denom else float("nan")}


def _class_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalars = [
        {k: v for k, v in row.items() if k not in ("record", "test_distances")}
        for row in rows
    ]
    summary = _mean_leaves(scalars)
    summary.update(_ratios(summary))
    return summary


def _split_summary(class_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary = _mean_leaves(list(class_summaries.values()))
    summary.update(_ratios(summary))
    summary["classes_rho_sel_ge_half"] = sum(
        1 for c in class_summaries.values() if c["rho_sel"] >= MIN_RHO_SEL
    )
    summary["classes"] = class_summaries
    return summary


def _guard_r(
    config: dict[str, Any], split_idx: int, split_summary: dict[str, Any]
) -> None:
    """Guard: our r_5 and r_ran20 must match exp-10's own census.json (r_5, r_random)."""
    exp10_summary = load_census(exp10_config(config))["splits"][str(split_idx)]
    for actual_key, expected_key in (("r_5", "r_5"), ("r_ran20", "r_random")):
        actual = float(split_summary[actual_key])
        expected = float(exp10_summary[expected_key])
        if abs(actual - expected) > _R_TOLERANCE:
            raise ValueError(
                f"Coverage {actual_key} mismatch against exp-10 census at split={split_idx}: "
                f"{actual} vs {expected}"
            )


def _split_quantities(
    config: dict[str, Any],
    split_idx: int,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    shared: _SplitInputs,
    class_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Every (arm, draw) cohort's similarity quantities for one split (secondary 4)."""
    ctx = _build_split_context(config, split_idx, full_df, train_df, *shared)
    cohorts: list[dict[str, Any]] = []
    for draw_idx in range(N_DRAWS):
        for allocation in ALLOCATIONS:
            per_class = {
                c: class_quantities(
                    ctx, c, class_rows[c][draw_idx]["record"][allocation]
                )
                for c in shared.class_names
            }
            cohorts.append(
                cohort_record(ctx, "shortage", 5, 32, allocation, draw_idx, per_class)
            )
    return cohorts


def _split_stage(
    config: dict[str, Any],
    split_idx: int,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    shared: _SplitInputs,
    split_allocations: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """One split's census summary, allocation record, and quantities cohorts."""
    embeddings = embed(shared.means, training_mu(shared.means, train_df))
    class_rows = _class_rows(full_df, embeddings, shared.class_names, split_allocations)
    class_summaries = {
        c_name: _class_summary(rows) for c_name, rows in class_rows.items()
    }
    split_summary = _split_summary(class_summaries)
    _guard_r(config, split_idx, split_summary)

    allocations = {
        "allocations": {
            str(d): {c: class_rows[c][d]["record"] for c in shared.class_names}
            for d in range(N_DRAWS)
        },
        "test_distances": {
            c_name: [row["test_distances"] for row in rows]
            for c_name, rows in class_rows.items()
        },
    }
    cohorts = _split_quantities(
        config, split_idx, full_df, train_df, shared, class_rows
    )
    return split_summary, allocations, cohorts


def _write_census_outputs(
    config: dict[str, Any],
    splits: dict[str, Any],
    all_allocations: dict[str, Any],
    cohorts: list[dict[str, Any]],
) -> Path:
    census_p = output_root(config) / "data" / "census.json"
    write_json(census_p, {"min_rho_sel": MIN_RHO_SEL, "splits": splits})
    sign_file(census_p)

    allocations_p = output_root(config) / "data" / "allocations.json"
    write_json(allocations_p, all_allocations)
    sign_file(allocations_p)

    quantities_p = output_root(config) / "data" / "quantities.json"
    write_json(quantities_p, {"cohorts": cohorts})
    sign_file(quantities_p)
    return census_p


def _load_inputs(
    config: dict[str, Any],
) -> tuple[
    list[str],
    dict[int, pd.DataFrame],
    dict[int, pd.DataFrame],
    _SplitInputs,
    dict[str, Any],
]:
    """Manifests, patient-class means, exp-6 ICC, and exp-10's own allocations."""
    class_names, full_dfs, train_dfs = _load_manifests(config)
    means, counts = patient_class_means(full_dfs[0])
    for s in range(1, N_SPLITS):
        assert_consistent_patch_counts(counts, full_dfs[s])
    rho_by_class, raw_full = _load_rho_bar(config, class_names)
    exp10_allocations = load_allocations(exp10_config(config))
    shared = _SplitInputs(means, class_names, rho_by_class, raw_full)
    return class_names, full_dfs, train_dfs, shared, exp10_allocations


def _gate(splits: dict[str, Any]) -> None:
    """Raise if any split's selection ratio rho_sel falls below the gate threshold."""
    failing = [s for s, v in splits.items() if v["rho_sel"] < MIN_RHO_SEL]
    if failing:
        raise RuntimeError(
            f"Selection ratio rho_sel below {MIN_RHO_SEL} for split(s) {failing}; "
            "redesign the shortage-selection rule against this census before fitting."
        )


def run_census(config: dict[str, Any]) -> Path:
    """Census the random/dispersed arms per split; write, sign, and gate on rho_sel.

    Writes before gating so a failing census is still inspectable, and raises
    afterwards so the downstream ``afterok`` fit array never starts.
    """
    class_names, full_dfs, train_dfs, shared, exp10_allocations = _load_inputs(config)

    splits: dict[str, Any] = {}
    all_allocations: dict[str, Any] = {}
    cohorts: list[dict[str, Any]] = []
    for s in range(N_SPLITS):
        split_summary, allocations, split_cohorts = _split_stage(
            config, s, full_dfs[s], train_dfs[s], shared, exp10_allocations[str(s)]
        )
        splits[str(s)] = split_summary
        all_allocations[str(s)] = allocations
        cohorts.extend(split_cohorts)

    census_p = _write_census_outputs(config, splits, all_allocations, cohorts)
    _gate(splits)
    return census_p
