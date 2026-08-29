from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from imbalance_benchmark.datasets.data import ImbalanceDataset
from imbalance_benchmark.datasets.features import load_feature_row
from imbalance_benchmark.analysis.predictors.signals.icc import (
    descriptive_support,
    icc_estimate,
)
from imbalance_benchmark.modeling.training.semantic_scale import _matched_draw_indices

__all__ = ["ICC_CASE_CAP", "ICC_PATCH_CAP", "build_descriptive_support"]

# Cost guard (report methods §sec:support-signal is descriptive only): a full
# feature pass over every condition's patches is too expensive, so the ICC
# estimate is drawn from a bounded sample. m_star/n_c use the exact case
# distribution, which costs nothing beyond the manifest already on disk.
ICC_CASE_CAP = 200
ICC_PATCH_CAP = 200


class _RawManifest:
    """Minimal ``.df``/``.classes`` stand-in avoiding ``ImbalanceDataset``'s eager load."""

    def __init__(self, df: pd.DataFrame, classes: list[str]) -> None:
        self.df = df
        self.classes = classes


def _load_rows(frame: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Load exactly the requested rows' raw features, bypassing the resident feature bank.

    The shared bank in ``datasets.features.cache`` never evicts, so building a
    full ``ImbalanceDataset`` per condition to sample a capped subset would
    still eagerly load and permanently retain every patch in that condition --
    exactly the full feature pass the cap exists to avoid.
    """
    paths = frame["feature_path"].astype(str).to_numpy()
    feature_index = (
        frame["feature_index"].to_numpy() if "feature_index" in frame else None
    )
    rows = [
        load_feature_row(
            str(paths[i]),
            int(feature_index[i])
            if feature_index is not None and pd.notna(feature_index[i])
            else None,
        )
        for i in indices
    ]
    return torch.stack(rows).numpy()


def _pca_direction(features: np.ndarray) -> np.ndarray:
    """Leading principal direction of a jointly-fit, class-balanced reference draw."""
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def _reference_direction(
    balanced_path: Path, class_names: list[str], seed: int
) -> np.ndarray:
    frame = pd.read_csv(balanced_path)
    manifest = cast(ImbalanceDataset, _RawManifest(frame, class_names))
    indices, _ = _matched_draw_indices(manifest, seed)
    features = _load_rows(frame, indices)
    return _pca_direction(features)


def _sample_condition_scores(
    class_frame: pd.DataFrame, direction: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Capped, seeded case/patch sample of one class's projected scalar scores."""
    frame = class_frame.reset_index(drop=True)
    if frame.empty:
        return np.array([]), np.array([])
    case_ids_all = frame["case_id"].to_numpy()
    unique_cases = np.unique(case_ids_all)
    chosen_cases = rng.choice(
        unique_cases, size=min(ICC_CASE_CAP, len(unique_cases)), replace=False
    )
    rows, case_ids = [], []
    for case in chosen_cases:
        case_rows = np.flatnonzero(case_ids_all == case)
        take = rng.choice(
            case_rows, size=min(ICC_PATCH_CAP, len(case_rows)), replace=False
        )
        rows.append(take)
        case_ids.append(np.full(len(take), case))
    indices = np.concatenate(rows)
    features = _load_rows(frame, indices)
    return features @ direction, np.concatenate(case_ids)


def _condition_descriptives(
    condition: dict[str, Any],
    class_names: list[str],
    direction: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    """Per-class descriptive ICC/m*/N_eff for one condition (report §sec:support-signal)."""
    frame = pd.read_csv(condition["path"])
    rng = np.random.default_rng(seed)
    out = {}
    for class_name in class_names:
        class_frame = frame.loc[frame["cancer_type"] == class_name]
        counts = class_frame["case_id"].value_counts().to_numpy()
        scores, case_ids = _sample_condition_scores(class_frame, direction, rng)
        out[class_name] = descriptive_support(counts, icc_estimate(scores, case_ids))
    return out


def _all_conditions(
    freeze: dict[str, Any], balanced: dict[str, Any], tail_assignments: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Every condition (balanced plus each assignment's realized conditions)."""
    return {
        "balanced": balanced,
        **{
            f"{assignment}_{severity}": condition
            for assignment in tail_assignments
            for severity, condition in freeze["assignment_conditions"][
                assignment
            ].items()
        },
    }


def build_descriptive_support(
    freeze: dict[str, Any],
    balanced: dict[str, Any],
    tail_assignments: dict[str, Any],
    class_names: list[str] | None,
    seed: int,
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Per-class ICC/m*/N_eff for every condition (balanced plus each assignment)."""
    names = class_names or list(balanced["allocated_counts"])
    direction = _reference_direction(
        Path(balanced["path"]), names, int(freeze["construction_seed"])
    )
    return {
        key: _condition_descriptives(condition, names, direction, seed)
        for key, condition in _all_conditions(
            freeze, balanced, tail_assignments
        ).items()
    }
