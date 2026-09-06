"""RQ-D3: R(method, a) per Eq. 5, reported only where the damage gate opened."""

from __future__ import annotations

from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.gates import (
    _SeverityInputs,
    _recovery_comparison,
)
from imbalance_benchmark.analysis.query import load_seed_predictions

from diversity.analyze.common import (
    CE,
    MATCHED_METHOD,
    N_PERMUTATIONS,
    N_REPLICATES,
    UNMATCHED_METHOD,
    endpoint_distribution,
    fixed_tail_classes,
    iter_splits,
)

__all__ = ["recovery_report"]


def _split_recovery(
    exp3_paths: dict[str, Any],
    freeze: dict[str, Any],
    allocation: str,
    endpoint: str,
    method: str,
    damage_entry: dict[str, Any],
    dataset: str,
) -> dict[str, Any] | None:
    class_names = list(freeze["class_names"])
    try:
        ce_narrow = load_seed_predictions(
            exp3_paths, allocation, CE, assignment="narrow"
        )
        method_narrow = load_seed_predictions(
            exp3_paths, allocation, method, assignment="narrow"
        )
    except RuntimeError:
        return None
    if ce_narrow is None or method_narrow is None:
        return None
    ctx = BootstrapContext(exp3_paths, False, N_REPLICATES, seed=0)
    tail_classes = fixed_tail_classes(freeze, class_names)
    ce_dist = endpoint_distribution(
        ctx, endpoint, ce_narrow, len(class_names), tail_classes
    )
    method_dist = endpoint_distribution(
        ctx, endpoint, method_narrow, len(class_names), tail_classes
    )
    if ce_dist is None or method_dist is None:
        return None
    numerator_dist = method_dist - ce_dist
    deficit_dist = np.asarray(damage_entry["bootstrap_effect"][: len(numerator_dist)])
    inp = _SeverityInputs(
        {},
        allocation,
        {},
        {},
        ctx,
        len(class_names),
        N_PERMUTATIONS,
        0,
        "narrow",
        dataset,
    )
    return _recovery_comparison(
        inp, method, endpoint, numerator_dist, deficit_dist, True, None
    )


def _method_recovery(
    config: dict[str, Any],
    dataset: str,
    allocation: str,
    endpoint: str,
    method: str,
    damage_entry: dict[str, Any],
) -> dict[str, Any] | None:
    per_split = [
        result
        for _, exp3_paths, freeze in iter_splits(config)
        if (
            result := _split_recovery(
                exp3_paths, freeze, allocation, endpoint, method, damage_entry, dataset
            )
        )
        is not None
    ]
    if not per_split:
        return None
    recoveries = [c["recovery"] for c in per_split if not np.isnan(c["recovery"])]
    return {
        "allocation": allocation,
        "endpoint": endpoint,
        "method": method,
        "recovery_mean": float(np.mean(recoveries)) if recoveries else float("nan"),
        "per_split": per_split,
    }


def recovery_report(
    config: dict[str, Any], dataset: str, damage: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """R(method, a) per Eq. 5, only in cells where the damage gate opened."""
    opened = {
        (d["allocation"], d["endpoint"]): d for d in damage if d.get("gate_passed")
    }
    reports = []
    for (allocation, endpoint), damage_entry in opened.items():
        for method in (UNMATCHED_METHOD, MATCHED_METHOD):
            report = _method_recovery(
                config, dataset, allocation, endpoint, method, damage_entry
            )
            if report is not None:
                reports.append(report)
    return reports
