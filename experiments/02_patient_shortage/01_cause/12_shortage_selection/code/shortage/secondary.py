"""Descriptive secondary analysis: selection gain within coverage-improvement tertiles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.analysis.query import load_test_identity

from breadth import N_SPLITS, exp2_split_paths

from sites.recall import PathsBySplit

from neighbours.census import load_allocations
from neighbours.secondary import _class_pairs, _draw_pairs, _tertile_rows

from composition.secondary import _pool_subgroup

from shortage import N_DRAWS

__all__ = ["secondary_analysis"]

TertileRows = tuple[list[Any], list[Any], list[Any]]


def _accumulate_class(
    pairs: dict[str, pd.DataFrame],
    c_idx: int,
    distances: dict[str, float],
    tertile_rows: TertileRows,
) -> None:
    """Fold one (split, draw, class) fit into the running tertile rows."""
    pairs_dis = _class_pairs(pairs["dispersed"], c_idx)
    pairs_ran = _class_pairs(pairs["random"], c_idx)
    if pairs_dis.empty:
        return
    for bucket, row in zip(
        tertile_rows, _tertile_rows(pairs_dis, pairs_ran, distances, low="dispersed")
    ):
        row["delta_sel"] = -row["gain"]
        bucket.append(row)


def _accumulate_split(
    config: dict[str, Any],
    paths12: PathsBySplit,
    s: int,
    class_names: list[str],
    cmap: dict[str, int],
    allocations: dict[str, Any],
    tertile_rows_by_class: dict[str, TertileRows],
) -> None:
    """Fold every draw of one split into the running per-class tertile rows."""
    manifest = exp2_split_paths(config, s)["data"] / "manifest.csv"
    case_ids = load_test_identity(manifest, is_mil=False)["case_id"].to_numpy()
    split_distances = allocations[str(s)]["test_distances"]
    for d in range(N_DRAWS):
        pairs = _draw_pairs(paths12, s, d, case_ids, ("dispersed", "random"))
        for c_name in class_names:
            _accumulate_class(
                pairs,
                cmap[c_name],
                split_distances[c_name][d],
                tertile_rows_by_class[c_name],
            )


def secondary_analysis(
    config: dict[str, Any],
    paths12: PathsBySplit,
    class_names: list[str],
    subgroup_idx: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Selection gain within tertiles of the test-patient coverage improvement.

    Each split's stored fits were labelled against that split's own frozen
    class order, so ``cmap`` is rebuilt per split rather than shared.
    """
    allocations = load_allocations(config)
    tertile_rows_by_class: dict[str, TertileRows] = {
        name: ([], [], []) for name in class_names
    }
    for s in range(N_SPLITS):
        split_class_names = list(
            load_freeze_meta(exp2_split_paths(config, s))["class_names"]
        )
        cmap = {name: i for i, name in enumerate(split_class_names)}
        _accumulate_split(
            config, paths12, s, class_names, cmap, allocations, tertile_rows_by_class
        )

    tertiles = {
        subgroup: _pool_subgroup([class_names[i] for i in idx], tertile_rows_by_class)
        for subgroup, idx in subgroup_idx.items()
    }
    return {"tertiles": tertiles}
