from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.common import sign_file, write_json
from imbalance_benchmark.datasets.data import ImbalanceDataset
from imbalance_benchmark.analysis.predictors.rq3_features import (
    _covariates,
    _deprived_classes,
    _reference_block,
)
from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    support_difficulty_alignment,
)
from imbalance_benchmark.analysis.predictors.signals.icc import (
    descriptive_support,
    icc_estimate,
)
from imbalance_benchmark.modeling.training.semantic_scale import _matched_draw_indices

__all__ = ["build_signal_profile", "write_signal_profile"]

logger = logging.getLogger(__name__)

# Cost guard (report methods §sec:support-signal is descriptive only): a full
# feature pass over every condition's patches is too expensive, so the ICC
# estimate is drawn from a bounded sample. m_star/n_c use the exact case
# distribution, which costs nothing beyond the manifest already on disk.
ICC_CASE_CAP = 200
ICC_PATCH_CAP = 200


def _nominal_shortage(balanced: dict[str, Any], imbalanced: dict[str, Any]) -> float:
    """Eq. (nominal-shortage): mean log loss of nominal allocation on deprived classes."""
    names = _deprived_classes(balanced, imbalanced)
    if not names:
        return 0.0
    bal, imb = balanced["allocated_counts"], imbalanced["allocated_counts"]
    return float(np.mean([np.log(bal[name] / imb[name]) for name in names]))


def _pca_direction(features: np.ndarray) -> np.ndarray:
    """Leading principal direction of a jointly-fit, class-balanced reference draw."""
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def _reference_direction(
    balanced_path: Path, class_names: list[str] | None, seed: int
) -> np.ndarray:
    dataset = ImbalanceDataset(balanced_path, class_names=class_names)
    indices, _ = _matched_draw_indices(dataset, seed)
    features = dataset.__getitems__(indices.tolist())["features"]
    return _pca_direction(np.asarray(features))


def _sample_condition_scores(
    dataset: ImbalanceDataset,
    class_name: str,
    direction: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Capped, seeded case/patch sample of one class's projected scalar scores."""
    class_column = dataset.df["cancer_type"].to_numpy()
    class_rows = np.flatnonzero(class_column == class_name)
    if len(class_rows) == 0:
        return np.array([]), np.array([])
    case_ids_all = dataset.df["case_id"].to_numpy()[class_rows]
    unique_cases = np.unique(case_ids_all)
    chosen_cases = rng.choice(
        unique_cases, size=min(ICC_CASE_CAP, len(unique_cases)), replace=False
    )
    rows, case_ids = [], []
    for case in chosen_cases:
        case_rows = class_rows[case_ids_all == case]
        take = rng.choice(
            case_rows, size=min(ICC_PATCH_CAP, len(case_rows)), replace=False
        )
        rows.append(take)
        case_ids.append(np.full(len(take), case))
    indices = np.concatenate(rows)
    features = np.asarray(dataset.__getitems__(indices.tolist())["features"])
    return features @ direction, np.concatenate(case_ids)


def _condition_descriptives(
    condition: dict[str, Any],
    class_names: list[str],
    direction: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    """Per-class descriptive ICC/m*/N_eff for one condition (report §sec:support-signal)."""
    dataset = ImbalanceDataset(condition["path"], class_names=class_names)
    rng = np.random.default_rng(seed)
    out = {}
    for class_name in class_names:
        counts = (
            dataset.df.loc[dataset.df["cancer_type"] == class_name, "case_id"]
            .value_counts()
            .to_numpy()
        )
        scores, case_ids = _sample_condition_scores(dataset, class_name, direction, rng)
        out[class_name] = descriptive_support(counts, icc_estimate(scores, case_ids))
    return out


def _build_comparisons(
    paths: dict[str, Path],
    freeze: dict[str, Any],
    is_mil: bool,
    class_names: list[str] | None,
    balanced: dict[str, Any],
    tail_assignments: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Per-(assignment, severity) shortage scores (protocol app:rq3 / app:testing)."""
    reference = _reference_block(
        paths, is_mil, class_names, balanced, int(freeze["construction_seed"])
    )
    comparisons = []
    for assignment in tail_assignments:
        for severity in ("moderate", "severe"):
            condition = freeze["assignment_conditions"][assignment][severity]
            shortages = _covariates(paths, is_mil, condition, reference, freeze)
            comparisons.append(
                {
                    "assignment": assignment,
                    "severity": severity,
                    "rho": condition["achieved_rho"],
                    "nominal_shortage": _nominal_shortage(balanced, condition),
                    "independent_shortage": shortages["independent_shortage"],
                    "diversity_shortage": shortages["diversity_shortage"],
                    "support_difficulty_alignment": support_difficulty_alignment(
                        condition, freeze
                    ),
                }
            )
    return comparisons


def _all_conditions(
    freeze: dict[str, Any], balanced: dict[str, Any], tail_assignments: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Every condition (balanced plus each assignment's moderate/severe)."""
    return {
        "balanced": balanced,
        **{
            f"{assignment}_{severity}": freeze["assignment_conditions"][assignment][
                severity
            ]
            for assignment in tail_assignments
            for severity in ("moderate", "severe")
        },
    }


def _build_descriptive_support(
    freeze: dict[str, Any],
    balanced: dict[str, Any],
    tail_assignments: dict[str, Any],
    class_names: list[str] | None,
    seed: int,
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Per-class ICC/m*/N_eff for every condition (balanced plus each assignment)."""
    names = class_names or list(balanced["allocated_counts"])
    direction = _reference_direction(
        Path(balanced["path"]), class_names, int(freeze["construction_seed"])
    )
    return {
        key: _condition_descriptives(condition, names, direction, seed)
        for key, condition in _all_conditions(
            freeze, balanced, tail_assignments
        ).items()
    }


def build_signal_profile(
    paths: dict[str, Path], freeze: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Pre-outcome signal profile: shortage scores and descriptive support diagnostics.

    Computed once per split from frozen, pre-mitigation evidence only, so RQ3
    and the matching rule (protocol app:testing) can share one source instead
    of recomputing the expensive diversity draw at every downstream step.
    """
    is_mil = False  # the matching protocol is scoped to patch classification
    class_names = list(freeze.get("class_names", [])) or None
    balanced = freeze["conditions"]["balanced"]
    tail_assignments = freeze.get("tail_assignments", {"native": []})
    comparisons = _build_comparisons(
        paths, freeze, is_mil, class_names, balanced, tail_assignments
    )
    descriptive_support = _build_descriptive_support(
        freeze, balanced, tail_assignments, class_names, seed
    )
    return {
        "comparisons": comparisons,
        "descriptive_support": descriptive_support,
        "icc_sampling_caps": {
            "cases_per_class": ICC_CASE_CAP,
            "patches_per_case": ICC_PATCH_CAP,
        },
        "freeze_content_sha256": freeze.get("content_sha256"),
    }


def write_signal_profile(
    paths: dict[str, Path], freeze: dict[str, Any], seed: int
) -> Path:
    """Build, write, and sign one split's ``signal_profile.json``."""
    profile = build_signal_profile(paths, freeze, seed)
    path = paths["data"] / "signal_profile.json"
    write_json(path, profile)
    sign_file(path)
    return path
