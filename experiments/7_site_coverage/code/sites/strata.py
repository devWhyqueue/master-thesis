"""Descriptive core/added/other site-gain breakdown of the broad5 -> broad10 step."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pandas as pd
from imbalance_benchmark.analysis.query import load_test_identity, read_run_record

from breadth import N_DRAWS, N_SPLITS, exp2_split_paths
from breadth.analyze.diagnostics import (
    _mean_leaves,
    _stratum_summary,
    patient_class_pairs,
)

from sites import allocation_dir
from sites.recall import PathsBySplit

__all__ = ["strata_analysis"]


def _site_strata_summary(
    pairs: pd.DataFrame, core_sites: set[str], added_sites: set[str]
) -> dict[str, float]:
    """Site-gain-relevant recall in the core, added, and other site strata."""
    is_core = pairs["site"].isin(list(core_sites)).to_numpy()
    is_added = pairs["site"].isin(list(added_sites)).to_numpy()
    core, added = _stratum_summary(pairs, is_core), _stratum_summary(pairs, is_added)
    other_mask = ~(is_core | is_added)
    weights = pairs["weight"].to_numpy()
    recall = pairs["recall"].to_numpy() * 100.0
    other_share = float(weights[other_mask].sum())
    other_recall = (
        float(np.average(recall[other_mask], weights=weights[other_mask]))
        if other_share > 0
        else float("nan")
    )
    return {
        "core_share": core["seen_share"],
        "core_recall": core["seen_recall"],
        "added_share": added["seen_share"],
        "added_recall": added["seen_recall"],
        "other_share": other_share,
        "other_recall": other_recall,
    }


def _draw_allocation_pairs(
    paths7: PathsBySplit, s: int, d: int, case_ids: np.ndarray
) -> dict[str, pd.DataFrame]:
    """Patient-class pairs for broad5 and broad10 at one (split, draw)."""
    pairs = {}
    for allocation in ("broad5", "broad10"):
        result_dir = allocation_dir(paths7[s], allocation, d)
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


def _class_strata_summary(
    pairs: dict[str, pd.DataFrame], record: dict[str, list[str]], c_idx: int
) -> dict[str, dict[str, float]]:
    """Core/added/other summary for broad5 and broad10, for one class/draw."""
    core, added = set(record["core"]), set(record["added"])
    summary = {}
    for allocation in ("broad5", "broad10"):
        class_pairs = pairs[allocation]
        class_pairs = cast(
            pd.DataFrame, class_pairs[class_pairs["label"] == c_idx]
        ).copy()
        if class_pairs.empty:
            continue
        class_pairs["weight"] = 1.0 / len(class_pairs)
        summary[allocation] = _site_strata_summary(class_pairs, core, added)
    return summary


def _split_summaries(
    config: dict[str, Any],
    paths7: PathsBySplit,
    s: int,
    site_classes: list[str],
    cmap: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Per-class strata summaries for every draw of one split."""
    manifest = exp2_split_paths(config, s)["data"] / "manifest.csv"
    case_ids = load_test_identity(manifest, is_mil=False)["case_id"].to_numpy()
    out: dict[str, list[dict[str, Any]]] = {c: [] for c in site_classes}
    for d in range(N_DRAWS):
        record_p = allocation_dir(paths7[s], "broad5", d) / "sites.json"
        site_record = json.loads(record_p.read_text(encoding="utf-8"))
        pairs = _draw_allocation_pairs(paths7, s, d, case_ids)
        for c in site_classes:
            if c not in site_record:
                continue
            summary = _class_strata_summary(pairs, site_record[c], cmap[c])
            if summary:
                out[c].append(summary)
    return out


def strata_analysis(
    config: dict[str, Any],
    paths7: PathsBySplit,
    site_classes: list[str],
    class_names: list[str],
) -> dict[str, Any]:
    """Descriptive core/added/other site breakdown, averaged over split-draws."""
    cmap = {name: i for i, name in enumerate(class_names)}
    per_class: dict[str, list[dict[str, Any]]] = {c: [] for c in site_classes}
    for s in range(N_SPLITS):
        for c, rows in _split_summaries(config, paths7, s, site_classes, cmap).items():
            per_class[c].extend(rows)
    return {c: _mean_leaves(rows) for c, rows in per_class.items() if rows}
