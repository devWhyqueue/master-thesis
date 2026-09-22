from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.common import sign_file, write_json
from imbalance_benchmark.analysis.predictors.rq3_features import (
    _covariates,
    _deprived_classes,
    _reference_block,
)
from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    support_difficulty_alignment,
)
from imbalance_benchmark.manifest.construction_helpers import CONDITION_REFERENCE
from imbalance_benchmark.analysis.predictors.signals.descriptive_support_profile import (
    ICC_CASE_CAP,
    ICC_PATCH_CAP,
    build_descriptive_support,
)

__all__ = ["build_signal_profile", "write_signal_profile"]

logger = logging.getLogger(__name__)


def _nominal_shortage(
    balanced: dict[str, Any], imbalanced: dict[str, Any], difficulty: dict[str, float]
) -> float:
    """Eq. (nominal-shortage): difficulty-weighted mean log loss on deprived classes.

    The allocated-count multiset is identical across tail assignments and
    only permuted onto different classes, so an unweighted mean cannot see
    which classes were deprived -- every assignment produces the same score.
    Weighting each deprived class's log loss by its frozen difficulty makes
    the score respond to which classes the assignment actually deprives.
    """
    names = _deprived_classes(balanced, imbalanced)
    if not names:
        return 0.0
    bal, imb = balanced["allocated_counts"], imbalanced["allocated_counts"]
    losses = np.array([np.log(bal[name] / imb[name]) for name in names])
    weights = np.array([difficulty[name] for name in names])
    if weights.sum() <= 0:
        return float(np.mean(losses))
    return float(np.average(losses, weights=weights))


def _build_comparisons(
    paths: dict[str, Path],
    freeze: dict[str, Any],
    is_mil: bool,
    class_names: list[str] | None,
    balanced: dict[str, Any],
    tail_assignments: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Per-(assignment, severity) shortage scores (protocol app:rq3 / app:testing)."""
    comparisons = []
    for assignment in tail_assignments:
        for severity in freeze["assignment_conditions"][assignment]:
            if severity == "balanced_spread":
                continue
            condition = freeze["assignment_conditions"][assignment][severity]
            reference_condition = CONDITION_REFERENCE[severity]
            reference_metadata = (
                freeze["assignment_conditions"][assignment][reference_condition]
                if reference_condition == "balanced_spread"
                else balanced
            )
            reference = _reference_block(
                paths,
                is_mil,
                class_names,
                reference_metadata,
                int(freeze["construction_seed"]),
            )
            shortages = _covariates(paths, is_mil, condition, reference, freeze)
            comparisons.append(
                {
                    "assignment": assignment,
                    "severity": severity,
                    "rho": condition["achieved_rho"],
                    "nominal_shortage": _nominal_shortage(
                        reference_metadata,
                        condition,
                        freeze.get("difficulty_evidence", {}).get("difficulty", {}),
                    ),
                    "independent_shortage": shortages["independent_shortage"],
                    "diversity_shortage": shortages["diversity_shortage"],
                    "support_difficulty_alignment": support_difficulty_alignment(
                        condition, freeze
                    ),
                }
            )
    return comparisons


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
    descriptive_support = build_descriptive_support(
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
