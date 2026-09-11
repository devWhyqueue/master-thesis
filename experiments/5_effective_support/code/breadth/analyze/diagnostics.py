"""Exploratory diagnostics: ICC sample census and tissue-source-site coverage.

Neither diagnostic is prespecified. The census documents how far the ICC
sampling caps bind; the site coverage asks whether broader patient sampling
helps TCGA-UT because it adds tissue source sites to the training allocation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.analysis.query import load_test_identity, read_run_record
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths, write_json

from breadth import GRID_CELLS, N_DRAWS, N_SPLITS, draw_dir, exp2_split_paths
from breadth.icc import ICC_CASE_CAP, ICC_PATCH_CAP
from breadth.sampling import sample_cell_draw

__all__ = [
    "equal_budget_decomposition",
    "icc_sample_census",
    "patient_class_pairs",
    "run_diagnostics",
    "tissue_source_site",
]

logger = logging.getLogger(__name__)

TCGA_BARCODE = re.compile(r"^TCGA-([A-Z0-9]{2})-")
# (site seen in the broad allocation, site seen in the deep allocation)
TRANSITIONS: dict[str, tuple[bool, bool]] = {
    "seen_both": (True, True),
    "gained": (True, False),
    "lost": (False, True),
    "unseen_both": (False, False),
}

CellRows = dict[tuple[int, int], list[dict[str, float]]]
SiteCoverage = tuple[CellRows, list[dict[str, Any]]]


def tissue_source_site(case_id: str) -> str | None:
    """Return the tissue source site code of a TCGA participant barcode."""
    match = TCGA_BARCODE.match(case_id)
    return match.group(1) if match else None


def icc_sample_census(
    train_df: pd.DataFrame, class_names: list[str]
) -> dict[str, dict[str, float]]:
    """Patients, capped sample size, and patch-cap exceedance per class."""
    census = {}
    for name in class_names:
        counts = train_df.loc[train_df["cancer_type"] == name, "case_id"].value_counts()
        census[name] = {
            "patients": int(len(counts)),
            "sampled_patients": int(min(len(counts), ICC_CASE_CAP)),
            "patients_above_patch_cap": int((counts > ICC_PATCH_CAP).sum()),
            "median_patches_per_patient": float(counts.median()),
        }
    return census


def patient_class_pairs(
    case_ids: np.ndarray, labels: np.ndarray, preds: np.ndarray
) -> pd.DataFrame:
    """Within-patient recalls with the weights of the patient-macro endpoint.

    The weights ``1 / (K |P_c|)`` sum to one, so the weighted recall sum is the
    patient-macro balanced accuracy of the draw.
    """
    frame = pd.DataFrame(
        {"case_id": case_ids.astype(str), "label": labels, "correct": preds == labels}
    )
    grouped = frame.groupby(["label", "case_id"], sort=True)
    pairs = grouped.agg(recall=("correct", "mean")).reset_index()
    class_size = pairs.groupby("label")["case_id"].transform("size")
    pairs["weight"] = 1.0 / (pairs["label"].nunique() * class_size)
    pairs["site"] = pairs["case_id"].map(tissue_source_site)
    return pairs


def _site_seen(
    pairs: pd.DataFrame, sample_df: pd.DataFrame, class_names: list[str]
) -> tuple[np.ndarray, float]:
    """Flag test pairs whose site trains the same class; count sites per class."""
    cmap = {name: i for i, name in enumerate(class_names)}
    sites = pd.DataFrame(
        {
            "label": [cmap[c] for c in sample_df["cancer_type"]],
            "site": sample_df["case_id"].astype(str).map(tissue_source_site),
        }
    ).drop_duplicates()
    merged = pairs[["label", "site"]].merge(
        sites, on=["label", "site"], how="left", indicator=True
    )
    seen = merged["_merge"].eq("both").to_numpy()
    return seen, float(sites.groupby("label").size().mean())


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean, NaN for an empty stratum."""
    return float(np.average(values, weights=weights)) if weights.sum() > 0 else np.nan


def _stratum_summary(pairs: pd.DataFrame, seen: np.ndarray) -> dict[str, float]:
    """Seen-site share and patient recall (%) of seen and unseen sites."""
    weights = pairs["weight"].to_numpy()
    recall = pairs["recall"].to_numpy() * 100.0
    return {
        "seen_share": float(weights[seen].sum()),
        "seen_recall": _weighted_mean(recall[seen], weights[seen]),
        "unseen_recall": _weighted_mean(recall[~seen], weights[~seen]),
    }


def equal_budget_decomposition(
    broad: pd.DataFrame,
    broad_seen: np.ndarray,
    deep: pd.DataFrame,
    deep_seen: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Split X into additive contributions of site-coverage transitions (points)."""
    weights = broad["weight"].to_numpy()
    delta = (broad["recall"].to_numpy() - deep["recall"].to_numpy()) * 100.0 * weights
    out = {}
    for name, (in_broad, in_deep) in TRANSITIONS.items():
        mask = (broad_seen == in_broad) & (deep_seen == in_deep)
        out[name] = {
            "share": float(weights[mask].sum()),
            "contribution": float(delta[mask].sum()),
        }
    return out


def _mean_leaves(items: list[Any]) -> Any:
    """Average nested dictionaries of floats key by key, ignoring NaN."""
    first = items[0]
    if not isinstance(first, dict):
        return float(np.nanmean(items))
    return {key: _mean_leaves([item[key] for item in items]) for key in first}


def _split_site_coverage(
    config: dict[str, Any],
    split_idx: int,
    train_df: pd.DataFrame,
    class_names: list[str],
    test_cases: np.ndarray,
) -> SiteCoverage:
    """Per-draw site summaries of every cell and the equal-budget decomposition."""
    paths = split_paths(ensure_dirs(config), split_idx)
    cells: CellRows = {c: [] for c in GRID_CELLS}
    decompositions = []
    for draw_idx in range(N_DRAWS):
        flagged = {}
        for g, m in GRID_CELLS:
            record = read_run_record(
                draw_dir(paths, g, m, draw_idx),
                splits=("test",),
                array_fields=("labels", "preds"),
            )
            if record is None:
                raise RuntimeError(f"Missing run record for G={g}, m={m}")
            test = record["splits"]["test"]
            pairs = patient_class_pairs(
                test_cases, np.asarray(test["labels"]), np.asarray(test["preds"])
            )
            sample_df = sample_cell_draw(
                train_df, class_names, g, m, split_idx, draw_idx
            )
            seen, n_sites = _site_seen(pairs, sample_df, class_names)
            cells[(g, m)].append({**_stratum_summary(pairs, seen), "sites": n_sites})
            flagged[(g, m)] = (pairs, seen)
        decompositions.append(
            equal_budget_decomposition(*flagged[(20, 8)], *flagged[(5, 32)])
        )
    return cells, decompositions


def _split_diagnostics(
    config: dict[str, Any], split_idx: int, class_names: list[str]
) -> tuple[dict[str, dict[str, float]], SiteCoverage | None]:
    """ICC census of one split and, for TCGA barcodes, its site coverage."""
    manifest = exp2_split_paths(config, split_idx)["data"] / "manifest.csv"
    train_df = pd.read_csv(manifest).query("split == 'train'").reset_index(drop=True)
    census = icc_sample_census(train_df, class_names)
    test_cases = load_test_identity(manifest, is_mil=False)["case_id"].to_numpy()
    if not all(tissue_source_site(c) for c in np.unique(test_cases)):
        return census, None
    logger.info("Site coverage for split %d", split_idx)
    return census, _split_site_coverage(
        config, split_idx, train_df, class_names, test_cases
    )


def run_diagnostics(config: dict[str, Any]) -> Path:
    """Write ``data/diagnostics.json`` for one dataset."""
    class_names = list(load_freeze_meta(exp2_split_paths(config, 0))["class_names"])
    payload: dict[str, Any] = {"icc_sample": {}}
    cells: CellRows = {c: [] for c in GRID_CELLS}
    decompositions: list[dict[str, Any]] = []
    for split_idx in range(N_SPLITS):
        census, coverage = _split_diagnostics(config, split_idx, class_names)
        payload["icc_sample"][str(split_idx)] = census
        if coverage is None:
            continue
        for cell, rows in coverage[0].items():
            cells[cell].extend(rows)
        decompositions.extend(coverage[1])
    if decompositions:
        payload["site_coverage"] = {
            "cells": {f"G{g}_m{m}": _mean_leaves(r) for (g, m), r in cells.items()},
            "equal_budget_decomposition": _mean_leaves(decompositions),
        }
    out_path = output_root(config) / "data" / "diagnostics.json"
    write_json(out_path, payload)
    return out_path
