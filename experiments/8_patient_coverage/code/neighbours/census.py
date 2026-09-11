"""Census stage: patient-mean embeddings, allocation draws, and the coverage gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.common import (
    output_root,
    sign_file,
    verify_signed_file,
    write_json,
)

from breadth import N_SPLITS, exp2_split_paths

from neighbours import MAX_LEAKAGE
from neighbours.coverage import split_census
from neighbours.embedding import (
    assert_consistent_patch_counts,
    embed,
    patient_class_means,
    training_mu,
)

__all__ = ["run_census", "load_census", "load_allocations"]


def _manifest(config: dict[str, Any], split_idx: int) -> pd.DataFrame:
    return pd.read_csv(exp2_split_paths(config, split_idx)["data"] / "manifest.csv")


def _load_manifests(
    config: dict[str, Any],
) -> tuple[list[str], dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Class names, full manifests, and their train-only partitions per split."""
    class_names = list(load_freeze_meta(exp2_split_paths(config, 0))["class_names"])
    full_dfs = {s: _manifest(config, s) for s in range(N_SPLITS)}
    train_dfs = {
        s: df.query("split == 'train'").reset_index(drop=True)
        for s, df in full_dfs.items()
    }
    return class_names, full_dfs, train_dfs


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
        split_summary, allocations = split_census(
            full_dfs[s], train_dfs[s], embeddings, class_names, s
        )
        splits[str(s)] = split_summary
        all_allocations[str(s)] = allocations
    return splits, all_allocations


def _write_census_outputs(
    config: dict[str, Any], splits: dict[str, Any], all_allocations: dict[str, Any]
) -> Path:
    payload = {
        "max_leakage": MAX_LEAKAGE,
        "splits": {
            s: {
                key: v[key]
                for key in ("r_deep", "r_neighbours", "r_random", "kappa", "classes")
            }
            for s, v in splits.items()
        },
    }
    census_p = output_root(config) / "data" / "census.json"
    write_json(census_p, payload)
    sign_file(census_p)

    allocations_p = output_root(config) / "data" / "allocations.json"
    write_json(allocations_p, all_allocations)
    sign_file(allocations_p)
    return census_p


def run_census(config: dict[str, Any]) -> Path:
    """Census the patient-coverage pool per split; write, sign, and gate on kappa.

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

    failing = [s for s, v in splits.items() if v["kappa"] > MAX_LEAKAGE]
    if failing:
        raise RuntimeError(
            f"Coverage leakage kappa exceeds {MAX_LEAKAGE} for split(s) {failing}; "
            "redesign the neighbour rule against this census before fitting."
        )
    return census_p


def load_census(config: dict[str, Any]) -> dict[str, Any]:
    """Load and verify the signed patient-coverage census."""
    census_p = output_root(config) / "data" / "census.json"
    verify_signed_file(census_p)
    return json.loads(census_p.read_text(encoding="utf-8"))


def load_allocations(config: dict[str, Any]) -> dict[str, Any]:
    """Load and verify the signed per-split, per-draw patient allocation records."""
    allocations_p = output_root(config) / "data" / "allocations.json"
    verify_signed_file(allocations_p)
    return json.loads(allocations_p.read_text(encoding="utf-8"))
