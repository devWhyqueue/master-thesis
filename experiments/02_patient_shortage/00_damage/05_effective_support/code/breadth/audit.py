"""Preflight audit: grid eligibility verification, ICC estimation, and freeze."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.common import output_root, sign_file
from imbalance_benchmark.datasets.features import load_feature_row

from breadth import (
    BREADTH_LADDER,
    DEPTH_LADDER,
    INPUT_DIM,
    N_SPLITS,
    exp2_split_paths,
)
from breadth.icc import (
    cell_effective_support,
    compute_class_icc,
    pca_leading_direction,
)
from breadth.sampling import check_grid_eligibility

__all__ = ["run_preflight"]

logger = logging.getLogger(__name__)


def _load_split_manifests(
    config: dict[str, Any],
) -> tuple[dict[int, pd.DataFrame], list[str], dict[str, Any]]:
    """Load train manifests and class names for all splits."""
    exp2_p0 = exp2_split_paths(config, 0)
    freeze = load_freeze_meta(exp2_p0)
    class_names = list(freeze["class_names"])

    train_dfs: dict[int, pd.DataFrame] = {}
    for split_idx in range(N_SPLITS):
        p = exp2_split_paths(config, split_idx)
        m_file = p["data"] / "manifest.csv"
        if not m_file.exists():
            raise FileNotFoundError(f"Missing manifest for split {split_idx}: {m_file}")
        df = pd.read_csv(m_file)
        train_dfs[split_idx] = cast(
            pd.DataFrame, df[df["split"] == "train"]
        ).reset_index(drop=True)

    return train_dfs, class_names, freeze


def _extract_reference_features(
    train_df: pd.DataFrame, class_names: list[str], seed: int
) -> np.ndarray:
    """Build class-balanced sample to find PCA leading direction."""
    rng = np.random.default_rng(seed)
    min_per_class = min(
        len(train_df[train_df["cancer_type"] == c]) for c in class_names
    )
    ref_n = min(100, min_per_class) if min_per_class > 0 else 0
    ref_rows = []
    for c in class_names:
        sub = train_df[train_df["cancer_type"] == c]
        ref_rows.extend(rng.choice(sub.index.to_numpy(), size=ref_n, replace=False))
    return np.stack(
        [
            load_feature_row(
                str(train_df.loc[i, "feature_path"]),
                int(train_df.loc[i, "feature_index"])
                if "feature_index" in train_df
                and pd.notna(train_df.loc[i, "feature_index"])
                else None,
            )
            for i in ref_rows
        ]
    )


def _compute_split_iccs(
    train_df: pd.DataFrame,
    class_names: list[str],
    seed: int = 0,
) -> dict[str, float]:
    """Compute ICC per class for one split using PCA reference projection."""
    ref_features = _extract_reference_features(train_df, class_names, seed)
    direction = pca_leading_direction(ref_features)
    rng = np.random.default_rng(seed)
    class_iccs = {}

    for c in class_names:
        sub = train_df[train_df["cancer_type"] == c].reset_index(drop=True)
        feat_rows = np.stack(
            [
                load_feature_row(
                    str(sub.loc[i, "feature_path"]),
                    int(sub.loc[i, "feature_index"])
                    if "feature_index" in sub and pd.notna(sub.loc[i, "feature_index"])
                    else None,
                )
                for i in range(len(sub))
            ]
        )
        case_ids = np.asarray(sub["case_id"].astype(str))
        class_iccs[c] = compute_class_icc(feat_rows, case_ids, direction, rng)
    return class_iccs


def _audit_all_split_iccs(
    train_dfs: dict[int, pd.DataFrame], class_names: list[str]
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Compute per-split and cohort-mean ICCs for all classes."""
    split_iccs: dict[str, dict[str, float]] = {}
    for s_idx in range(N_SPLITS):
        logger.info("Computing ICCs for split %d", s_idx)
        split_iccs[str(s_idx)] = _compute_split_iccs(
            train_dfs[s_idx], class_names, seed=s_idx
        )
    cohort_iccs = {
        c: float(np.mean([split_iccs[str(s)][c] for s in range(N_SPLITS)]))
        for c in class_names
    }
    return split_iccs, cohort_iccs


def _assemble_report(
    class_names: list[str],
    real_b: tuple[int, ...],
    real_d: tuple[int, ...],
    audit: dict[str, Any],
    split_iccs: dict[str, dict[str, float]],
    cohort_iccs: dict[str, float],
) -> dict[str, Any]:
    """Assemble preflight audit report dict."""
    neff_by_cell = {
        f"G{g}_m{m}": cell_effective_support(g, m, cohort_iccs)
        for g in real_b
        for m in real_d
    }
    return {
        "status": "pass",
        "input_dim": INPUT_DIM,
        "class_names": class_names,
        "breadth_ladder": list(real_b),
        "depth_ladder": list(real_d),
        "counts_audit": audit,
        "split_iccs": split_iccs,
        "cohort_iccs": cohort_iccs,
        "cell_effective_supports": neff_by_cell,
    }


def run_preflight(config: dict[str, Any]) -> Path:
    """Audit patient eligibility across splits and compute class ICCs."""
    train_dfs, class_names, _ = _load_split_manifests(config)
    real_b, real_d, audit = check_grid_eligibility(
        train_dfs, class_names, BREADTH_LADDER, DEPTH_LADDER
    )
    split_iccs, cohort_iccs = _audit_all_split_iccs(train_dfs, class_names)
    report = _assemble_report(
        class_names, real_b, real_d, audit, split_iccs, cohort_iccs
    )
    out_p = output_root(config) / "data" / "preflight.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sign_file(out_p)
    logger.info("Preflight signed at %s", out_p)
    return out_p
