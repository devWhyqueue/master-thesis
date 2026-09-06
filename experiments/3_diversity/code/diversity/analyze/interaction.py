"""RQ-D2: the interaction D_div(severe) - D_div(balanced), per endpoint."""

from __future__ import annotations

from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.gates import confidence_interval
from imbalance_benchmark.analysis.inference.permutation import _contribution_p_value

from diversity.analyze.combine import fisher_combine
from diversity.analyze.common import ENDPOINTS, N_PERMUTATIONS

__all__ = ["interaction_report"]


def _split_interaction_p(severe_vec: np.ndarray, balanced_vec: np.ndarray) -> float:
    n = min(len(severe_vec), len(balanced_vec))
    diff = (severe_vec[:n] - balanced_vec[:n]).copy()
    observed = float(diff.sum())
    diff[-1] += observed - diff.sum()
    return _contribution_p_value(diff, observed, N_PERMUTATIONS, seed=0)


def _endpoint_interaction(
    severe: dict[str, Any], balanced: dict[str, Any], endpoint: str
) -> dict[str, Any]:
    per_split_p = [
        _split_interaction_p(s, b)
        for s, b in zip(
            severe["contribution_vectors"], balanced["contribution_vectors"]
        )
    ]
    severe_boot, balanced_boot = (
        np.asarray(severe["bootstrap_effect"]),
        np.asarray(balanced["bootstrap_effect"]),
    )
    m = min(len(severe_boot), len(balanced_boot))
    diff_boot = severe_boot[:m] - balanced_boot[:m]
    return {
        "endpoint": endpoint,
        "effect": float(diff_boot[0]),
        "ci": confidence_interval(diff_boot),
        "p_value": fisher_combine(per_split_p),
        "status": "tested",
    }


def interaction_report(damage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """D_div(severe) - D_div(balanced), per endpoint, via differenced patient contributions."""
    by_key = {
        (d["allocation"], d["endpoint"]): d
        for d in damage
        if "contribution_vectors" in d
    }
    reports = []
    for endpoint in ENDPOINTS:
        severe, balanced = (
            by_key.get(("severe", endpoint)),
            by_key.get(("balanced", endpoint)),
        )
        if severe is None or balanced is None:
            reports.append(
                {
                    "endpoint": endpoint,
                    "status": "not tested",
                    "reason": "damage contrast unavailable",
                }
            )
        else:
            reports.append(_endpoint_interaction(severe, balanced, endpoint))
    return reports
