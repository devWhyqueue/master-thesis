"""Census stage: clustered/random/dispersed draws and the coverage-ratio gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from imbalance_benchmark.common import output_root, sign_file, write_json

from breadth import N_SPLITS
from breadth.analyze.diagnostics import _mean_leaves

from neighbours.census import _load_manifests
from neighbours.embedding import (
    assert_consistent_patch_counts,
    embed,
    patient_class_means,
    training_mu,
)

from composition import MIN_RHO_CON, N_DRAWS
from composition.allocation import class_draw

__all__ = ["run_census"]


def _class_rows(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    embeddings: dict[tuple[str, str], Any],
    class_names: list[str],
    split: int,
) -> dict[str, list[dict[str, Any]]]:
    """Every class's five draws for one split."""
    class_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in class_names}
    for c_idx, c_name in enumerate(class_names):
        for draw_idx in range(N_DRAWS):
            class_rows[c_name].append(
                class_draw(
                    train_df,
                    full_df,
                    embeddings,
                    class_names,
                    c_idx,
                    c_name,
                    (split, draw_idx),
                )
            )
    return class_rows


def _ratios(r: dict[str, float]) -> dict[str, float]:
    """Coverage ratios rho_con and rho_sel (Eq. ratios)."""
    denom = r["r_5"] - r["r_random"]
    return {
        "rho_con": (r["r_clustered"] - r["r_random"]) / denom
        if denom
        else float("nan"),
        "rho_sel": (r["r_random"] - r["r_dispersed"]) / denom
        if denom
        else float("nan"),
    }


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
    summary["classes_rho_con_ge_half"] = sum(
        1 for c in class_summaries.values() if c["rho_con"] >= MIN_RHO_CON
    )
    summary["classes"] = class_summaries
    return summary


def _split_census(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    embeddings: dict[tuple[str, str], Any],
    class_names: list[str],
    split: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One split's per-class census and allocation records."""
    class_rows = _class_rows(train_df, full_df, embeddings, class_names, split)
    class_summaries = {
        c_name: _class_summary(rows) for c_name, rows in class_rows.items()
    }
    allocations = {
        "allocations": {
            str(d): {c_name: class_rows[c_name][d]["record"] for c_name in class_names}
            for d in range(N_DRAWS)
        },
        "test_distances": {
            c_name: [row["test_distances"] for row in rows]
            for c_name, rows in class_rows.items()
        },
    }
    return _split_summary(class_summaries), allocations


def _all_splits_census(
    class_names: list[str],
    full_dfs: dict[int, pd.DataFrame],
    train_dfs: dict[int, pd.DataFrame],
    means: dict[tuple[str, str], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every split's census summary and allocation records."""
    splits: dict[str, Any] = {}
    all_allocations: dict[str, Any] = {}
    for s in range(N_SPLITS):
        embeddings = embed(means, training_mu(means, train_dfs[s]))
        split_summary, allocations = _split_census(
            full_dfs[s], train_dfs[s], embeddings, class_names, s
        )
        splits[str(s)] = split_summary
        all_allocations[str(s)] = allocations
    return splits, all_allocations


def _write_census_outputs(
    config: dict[str, Any], splits: dict[str, Any], all_allocations: dict[str, Any]
) -> Path:
    census_p = output_root(config) / "data" / "census.json"
    write_json(census_p, {"min_rho_con": MIN_RHO_CON, "splits": splits})
    sign_file(census_p)

    allocations_p = output_root(config) / "data" / "allocations.json"
    write_json(allocations_p, all_allocations)
    sign_file(allocations_p)
    return census_p


def run_census(config: dict[str, Any]) -> Path:
    """Census the clustered/random/dispersed pool per split; write, sign, and gate.

    Writes before gating so a failing census is still inspectable, and raises
    afterwards so the downstream ``afterok`` fit array never starts.
    """
    class_names, full_dfs, train_dfs = _load_manifests(config)
    means, counts = patient_class_means(full_dfs[0])
    for s in range(1, N_SPLITS):
        assert_consistent_patch_counts(counts, full_dfs[s])

    splits, all_allocations = _all_splits_census(
        class_names, full_dfs, train_dfs, means
    )
    census_p = _write_census_outputs(config, splits, all_allocations)

    failing = [s for s, v in splits.items() if v["rho_con"] < MIN_RHO_CON]
    if failing:
        raise RuntimeError(
            f"Coverage ratio rho_con below {MIN_RHO_CON} for split(s) {failing}; "
            "redesign the cohort-composition rule against this census before fitting."
        )
    return census_p
