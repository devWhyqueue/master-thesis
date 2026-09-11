"""Descriptive secondary analyses: coverage-gain tertiles and class-level correlation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from imbalance_benchmark.analysis.query import load_test_identity, read_run_record
from scipy.stats import spearmanr

from breadth import N_DRAWS, N_SPLITS, exp2_split_paths
from breadth.analyze.diagnostics import _mean_leaves, patient_class_pairs

from neighbours import allocation_dir
from neighbours.accuracy import PathsBySplit
from neighbours.census import load_allocations, load_census

__all__ = ["secondary_analysis"]


def _draw_pairs(
    paths8: PathsBySplit, s: int, d: int, case_ids: np.ndarray
) -> dict[str, pd.DataFrame]:
    """Patient-class pairs for the neighbours and random allocations at one (split, draw)."""
    pairs = {}
    for allocation in ("neighbours", "random"):
        result_dir = allocation_dir(paths8[s], allocation, d)
        rec = read_run_record(
            result_dir, splits=("test",), array_fields=("labels", "preds")
        )
        if rec is None or "test" not in rec.get("splits", {}):
            raise RuntimeError(f"Missing run record at {result_dir}")
        test = rec["splits"]["test"]
        pairs[allocation] = patient_class_pairs(
            case_ids, np.asarray(test["labels"]), np.asarray(test["preds"])
        )
    return pairs


def _class_pairs(pairs: pd.DataFrame, c_idx: int) -> pd.DataFrame:
    class_pairs = cast(pd.DataFrame, pairs[pairs["label"] == c_idx]).copy()
    if not class_pairs.empty:
        class_pairs["weight"] = 1.0 / len(class_pairs)
    return class_pairs


def _tertile_rows(
    pairs_n: pd.DataFrame, pairs_r: pd.DataFrame, distances: dict[str, float]
) -> list[dict[str, float]]:
    """Near/middle/far tertile recall_neighbours, recall_random, and gain."""
    case_ids = pairs_n["case_id"].to_numpy()
    dist = np.array([distances[str(c)] for c in case_ids])
    order = np.argsort(dist, kind="stable")
    groups = np.array_split(order, 3)
    weight = pairs_n["weight"].to_numpy()
    recall_n = pairs_n["recall"].to_numpy() * 100.0
    recall_r_by_case = dict(zip(pairs_r["case_id"], pairs_r["recall"] * 100.0))
    recall_r = np.array([recall_r_by_case[c] for c in case_ids])
    rows = []
    for group in groups:
        w = weight[group]
        recall_neighbours = float(np.average(recall_n[group], weights=w))
        recall_random = float(np.average(recall_r[group], weights=w))
        rows.append(
            {
                "recall_neighbours": recall_neighbours,
                "recall_random": recall_random,
                "gain": recall_random - recall_neighbours,
            }
        )
    return rows


def _tertile_labels(rows: list[list[dict[str, float]]]) -> dict[str, Any]:
    return {
        name: _mean_leaves(bucket)
        for name, bucket in zip(("near", "middle", "far"), rows)
    }


def _class_level(
    class_gains: dict[str, list[float]], census: dict[str, Any], class_names: list[str]
) -> dict[str, Any]:
    """Spearman rank correlation of class-level Delta_C against r_N - r_R."""
    delta_c, coverage_diff = [], []
    for c_name in class_names:
        gains = class_gains[c_name]
        if not gains:
            continue
        r_n = np.mean(
            [
                census["splits"][str(s)]["classes"][c_name]["r_neighbours"]
                for s in range(N_SPLITS)
            ]
        )
        r_r = np.mean(
            [
                census["splits"][str(s)]["classes"][c_name]["r_random"]
                for s in range(N_SPLITS)
            ]
        )
        delta_c.append(float(np.mean(gains)))
        coverage_diff.append(float(r_n - r_r))
    rho, pvalue = cast(tuple[float, float], spearmanr(delta_c, coverage_diff))
    return {
        "n_classes": len(delta_c),
        "spearman_r": float(rho),
        "spearman_p": float(pvalue),
    }


def _accumulate_class(
    pairs: dict[str, pd.DataFrame],
    c_name: str,
    c_idx: int,
    distances: dict[str, float],
    class_gains: dict[str, list[float]],
    tertile_rows: tuple[list[Any], list[Any], list[Any]],
) -> None:
    """Fold one (split, draw, class) fit into its running gain and tertile rows."""
    pairs_n = _class_pairs(pairs["neighbours"], c_idx)
    pairs_r = _class_pairs(pairs["random"], c_idx)
    if pairs_n.empty:
        return
    recall_n = float(np.average(pairs_n["recall"] * 100.0, weights=pairs_n["weight"]))
    recall_r = float(np.average(pairs_r["recall"] * 100.0, weights=pairs_r["weight"]))
    class_gains[c_name].append(recall_r - recall_n)
    for bucket, row in zip(tertile_rows, _tertile_rows(pairs_n, pairs_r, distances)):
        bucket.append(row)


def _accumulate_split(
    config: dict[str, Any],
    paths8: PathsBySplit,
    s: int,
    class_names: list[str],
    cmap: dict[str, int],
    allocations: dict[str, Any],
    class_gains: dict[str, list[float]],
    tertile_rows: tuple[list[Any], list[Any], list[Any]],
) -> None:
    """Fold every draw of one split into the running gains and tertile rows."""
    manifest = exp2_split_paths(config, s)["data"] / "manifest.csv"
    case_ids = load_test_identity(manifest, is_mil=False)["case_id"].to_numpy()
    split_distances = allocations[str(s)]["test_distances"]
    for d in range(N_DRAWS):
        pairs = _draw_pairs(paths8, s, d, case_ids)
        for c_name in class_names:
            _accumulate_class(
                pairs,
                c_name,
                cmap[c_name],
                split_distances[c_name][d],
                class_gains,
                tertile_rows,
            )


def secondary_analysis(
    config: dict[str, Any], paths8: PathsBySplit, class_names: list[str]
) -> dict[str, Any]:
    """Coverage-gain tertiles and the class-level rank correlation (descriptive)."""
    cmap = {name: i for i, name in enumerate(class_names)}
    allocations = load_allocations(config)
    census = load_census(config)
    tertile_rows: tuple[list[Any], list[Any], list[Any]] = ([], [], [])
    class_gains: dict[str, list[float]] = {name: [] for name in class_names}
    for s in range(N_SPLITS):
        _accumulate_split(
            config, paths8, s, class_names, cmap, allocations, class_gains, tertile_rows
        )

    return {
        "tertiles": _tertile_labels(list(tertile_rows)),
        "class_level": _class_level(class_gains, census, class_names),
    }
