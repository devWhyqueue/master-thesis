"""RQ-D1: D_div(a) = M(wide,a) - M(narrow,a) on the CE arm, per allocation and endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.permutation import (
    _ba_patient_contributions,
    _tail_nll_patient_contributions,
    paired_block_permutation_ba,
    paired_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.query import load_seed_predictions

from diversity.analyze.combine import combine_splits
from diversity.analyze.common import (
    CE,
    ENDPOINTS,
    N_PERMUTATIONS,
    N_REPLICATES,
    endpoint_distribution,
    fixed_tail_classes,
    iter_splits,
)
from diversity.analyze.thresholds import gate_passes, gate_thresholds
from diversity.manifests import ALLOCATIONS

__all__ = ["SplitDamage", "damage_report"]


@dataclass(frozen=True)
class SplitDamage:
    """One split's wide-vs-narrow contrast: effect distribution, p-value, contributions."""

    effect_dist: (
        np.ndarray
    )  # index 0 observed, wide - narrow (eq. 4, literal, unsigned)
    p_value: float
    contribution_vector: (
        np.ndarray
    )  # additive per-patient contributions, same sign as effect


def _ba_contrast(
    narrow: dict[str, Any], wide: dict[str, Any], ctx: BootstrapContext, n_classes: int
) -> tuple[float, np.ndarray]:
    p_value = paired_block_permutation_ba(
        narrow["labels"],
        wide["preds"],
        narrow["preds"],
        ctx.case_ids,
        n_classes,
        N_PERMUTATIONS,
    )
    _, contrib = _ba_patient_contributions(
        narrow["labels"], wide["preds"], narrow["preds"], ctx.case_ids, n_classes
    )
    return float(p_value), contrib


def _tail_nll_contrast(
    narrow: dict[str, Any],
    wide: dict[str, Any],
    ctx: BootstrapContext,
    tail_classes: list[int],
) -> tuple[float, np.ndarray]:
    p_value = paired_block_permutation_tail_nll(
        narrow["labels"],
        wide["probs"],
        narrow["probs"],
        ctx.case_ids,
        tail_classes,
        N_PERMUTATIONS,
    )
    # ``_tail_nll_patient_contributions`` is ce-minus-method oriented
    # (positive when the method arm has *lower* NLL); negate so the sign
    # matches this module's wide-minus-narrow convention (eq. 4, literal).
    _, raw_contrib = _tail_nll_patient_contributions(
        narrow["labels"], wide["probs"], narrow["probs"], ctx.case_ids, tail_classes
    )
    return float(p_value), -raw_contrib


def _split_damage(
    exp3_paths: dict[str, Path],
    freeze: dict[str, Any],
    allocation: str,
    endpoint: str,
    class_names: list[str],
) -> SplitDamage | None:
    """One split's wide-vs-narrow damage contrast on the CE arm."""
    narrow = load_seed_predictions(exp3_paths, allocation, CE, assignment="narrow")
    wide = load_seed_predictions(exp3_paths, allocation, CE, assignment="wide")
    if narrow is None or wide is None:
        return None
    ctx = BootstrapContext(exp3_paths, False, N_REPLICATES, seed=0)
    n_classes = len(class_names)
    tail_classes = fixed_tail_classes(freeze, class_names)
    narrow_dist = endpoint_distribution(ctx, endpoint, narrow, n_classes, tail_classes)
    wide_dist = endpoint_distribution(ctx, endpoint, wide, n_classes, tail_classes)
    if narrow_dist is None or wide_dist is None:
        return None
    if endpoint == "ba":
        p_value, contrib = _ba_contrast(narrow, wide, ctx, n_classes)
    else:
        p_value, contrib = _tail_nll_contrast(narrow, wide, ctx, tail_classes)
    return SplitDamage(wide_dist - narrow_dist, p_value, contrib)


def _cell_report(
    config: dict[str, Any], allocation: str, endpoint: str, threshold: float
) -> dict[str, Any]:
    per_split = [
        result
        for _, exp3_paths, freeze in iter_splits(config)
        if (
            result := _split_damage(
                exp3_paths, freeze, allocation, endpoint, list(freeze["class_names"])
            )
        )
        is not None
    ]
    if not per_split:
        return {
            "allocation": allocation,
            "endpoint": endpoint,
            "status": "not tested",
            "reason": "no fitted splits",
        }
    combined = combine_splits(
        [r.effect_dist for r in per_split], [r.p_value for r in per_split]
    )
    gate_passed = gate_passes(combined["effect"], combined["ci"], threshold)
    return {
        "allocation": allocation,
        "endpoint": endpoint,
        "gate_threshold": threshold,
        "gate_passed": gate_passed,
        "status": "tested" if gate_passed else "not tested",
        "contribution_vectors": [r.contribution_vector for r in per_split],
        **combined,
    }


def damage_report(config: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    """One report per (allocation, endpoint): the four confirmatory damage tests."""
    thresholds = gate_thresholds(config, dataset)
    threshold_by_endpoint = {
        "ba": thresholds.discrimination_threshold,
        "tail_nll": thresholds.calibration_threshold,
    }
    return [
        _cell_report(config, allocation, endpoint, threshold_by_endpoint[endpoint])
        for allocation in ALLOCATIONS
        for endpoint in ENDPOINTS
    ]
