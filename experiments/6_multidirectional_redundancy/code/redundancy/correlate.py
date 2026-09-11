"""Correlate stage: single-direction and full-feature ICC on the identical sample."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.common import output_root, sign_file, verify_signed_file
from imbalance_benchmark.datasets.features import load_feature_row

from breadth import N_REPLICATES, N_SPLITS, exp2_split_paths
from breadth.audit import _extract_reference_features
from breadth.icc import (
    ICC_CASE_CAP,
    ICC_PATCH_CAP,
    pca_leading_direction,
    sample_class_indices,
)

from redundancy import MEASURES, TRAINING_BOOTSTRAP_SEED, exp5_config
from redundancy.estimator import ClusterStats, cluster_stats, weighted_icc

__all__ = ["run_correlate"]

logger = logging.getLogger(__name__)

_GUARD_TOLERANCE = 1e-4
SplitStats = dict[str, dict[str, ClusterStats]]


def _class_subframe(train_df: pd.DataFrame, class_name: str) -> pd.DataFrame:
    sub = train_df[train_df["cancer_type"] == class_name]
    return cast(pd.DataFrame, sub).reset_index(drop=True)


def _load_rows(sub: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Load only the sampled feature rows for one class."""
    has_feature_index = "feature_index" in sub.columns
    rows = []
    for i in indices:
        row = sub.iloc[int(i)]
        feature_index = None
        if has_feature_index and pd.notna(row["feature_index"]):
            feature_index = int(row["feature_index"])
        rows.append(load_feature_row(str(row["feature_path"]), feature_index))
    return np.stack(rows).astype(np.float64)


def _sample_split(
    train_df: pd.DataFrame, class_names: list[str], split_idx: int
) -> SplitStats:
    """Reproduce exp-5's per-split sample and build both measures' cluster stats."""
    direction = pca_leading_direction(
        _extract_reference_features(train_df, class_names, seed=split_idx)
    )
    rng = np.random.default_rng(split_idx)
    stats: SplitStats = {}
    for class_name in class_names:
        sub = _class_subframe(train_df, class_name)
        case_ids = np.asarray(sub["case_id"].astype(str))
        indices, sampled_cases = sample_class_indices(
            case_ids, rng, ICC_CASE_CAP, ICC_PATCH_CAP
        )
        features = _load_rows(sub, indices)
        scores = features @ direction
        stats[class_name] = {
            "single": cluster_stats(scores[:, np.newaxis], sampled_cases),
            "full": cluster_stats(features, sampled_cases),
        }
    return stats


def _load_train_manifest(config: dict[str, Any], split_idx: int) -> pd.DataFrame:
    paths = exp2_split_paths(config, split_idx)
    manifest = pd.read_csv(paths["data"] / "manifest.csv")
    train = cast(pd.DataFrame, manifest[manifest["split"] == "train"])
    return train.reset_index(drop=True)


def _draw_training_bootstrap_weights(n_patients: int) -> np.ndarray:
    """Row 0 is every patient once; rows 1.. are cluster-bootstrap multiplicities."""
    weights = np.ones((N_REPLICATES, n_patients), dtype=np.float64)
    if N_REPLICATES > 1 and n_patients > 0:
        rng = np.random.default_rng(TRAINING_BOOTSTRAP_SEED)
        weights[1:] = rng.multinomial(
            n_patients, np.full(n_patients, 1.0 / n_patients), size=N_REPLICATES - 1
        )
    return weights


def _verify_against_exp5_preflight(
    config: dict[str, Any], class_names: list[str], single_raw: np.ndarray
) -> None:
    """Guard: the reproduced sample's clipped single-direction ICC must match exp-5."""
    preflight_p = output_root(exp5_config(config)) / "data" / "preflight.json"
    verify_signed_file(preflight_p)
    preflight = json.loads(preflight_p.read_text(encoding="utf-8"))
    split_iccs = preflight["split_iccs"]
    clipped = np.clip(single_raw[0], 0.0, 1.0)
    for split_idx in range(N_SPLITS):
        for class_idx, class_name in enumerate(class_names):
            expected = float(split_iccs[str(split_idx)][class_name])
            actual = float(clipped[split_idx, class_idx])
            if abs(actual - expected) > _GUARD_TOLERANCE:
                raise ValueError(
                    "Reproduced sample does not match exp-5 preflight ICC at "
                    f"split={split_idx} class={class_name}: {actual} vs {expected}"
                )


def _build_patient_index(per_split_stats: dict[int, SplitStats]) -> dict[str, int]:
    """Union of distinct patients across every (split, class) sample, sorted."""
    global_patients = sorted(
        {
            case
            for split_stats in per_split_stats.values()
            for class_stats in split_stats.values()
            for case in class_stats["single"].cases
        }
    )
    return {case: i for i, case in enumerate(global_patients)}


def _collect_raw_iccs(
    per_split_stats: dict[int, SplitStats],
    class_names: list[str],
    patient_index: dict[str, int],
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Weighted ICC of every (measure, split, class) under every replicate weight."""
    raw = {
        measure: np.empty((N_REPLICATES, N_SPLITS, len(class_names)), dtype=np.float64)
        for measure in MEASURES
    }
    for split_idx, split_stats in per_split_stats.items():
        for class_idx, class_name in enumerate(class_names):
            for measure in MEASURES:
                stats = split_stats[class_name][measure]
                cols = [patient_index[case] for case in stats.cases]
                raw[measure][:, split_idx, class_idx] = weighted_icc(
                    stats, weights[:, cols]
                )
    return raw


def _write_correlations_npz(
    config: dict[str, Any], class_names: list[str], raw: dict[str, np.ndarray]
) -> Path:
    out_p = output_root(config) / "data" / "correlations.npz"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_p,
        single=raw["single"],
        full=raw["full"],
        class_names=np.asarray(class_names),
    )
    sign_file(out_p)
    return out_p


def run_correlate(config: dict[str, Any]) -> Path:
    """Compute both redundancy measures per split/class and write the signed npz."""
    class_names = list(load_freeze_meta(exp2_split_paths(config, 0))["class_names"])

    per_split_stats: dict[int, SplitStats] = {}
    for split_idx in range(N_SPLITS):
        logger.info("Sampling and computing cluster stats for split %d", split_idx)
        train_df = _load_train_manifest(config, split_idx)
        per_split_stats[split_idx] = _sample_split(train_df, class_names, split_idx)

    patient_index = _build_patient_index(per_split_stats)
    weights = _draw_training_bootstrap_weights(len(patient_index))
    raw = _collect_raw_iccs(per_split_stats, class_names, patient_index, weights)

    _verify_against_exp5_preflight(config, class_names, raw["single"])
    out_p = _write_correlations_npz(config, class_names, raw)
    logger.info("Correlate stage signed at %s", out_p)
    return out_p
