from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "PRIMARY_METHODS",
    "holm_adjust_pvalues",
    "confirmatory_family",
    "apply_holm",
]

# Report §"Imbalance deficit, recovery, and inference": one confirmatory method per
# signal (prevalence / nominal support / independent support / difficulty /
# diversity), all a mean-one class weight on unmodified CE, so a difference
# between members is attributable to the signal rather than the intervention
# mechanism. Patch-regime family (protocol report tab:roster). focal,
# logit_adjustment and balanced_sampling moved to exploratory.
PRIMARY_METHODS = frozenset(
    {
        "weighted_ce",
        "class_balanced_ce",
        "independent_support_ce",
        "pilot_difficulty_ce",
        "semantic_scale_ce",
    }
)


def holm_adjust_pvalues(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment (sort, cumulative max of p*(m-i), clip to 1)."""
    m = len(pvalues)
    if m == 0:
        return []
    idx = np.argsort(pvalues)
    adj = [0.0] * m
    prev = 0.0
    for i, s_idx in enumerate(idx):
        prev = max(prev, min(1.0, pvalues[s_idx] * (m - i)))
        adj[s_idx] = prev
    return adj


def confirmatory_family(
    comparisons: list[dict[str, Any]],
) -> tuple[list[dict], list[dict]]:
    """Split gate comparisons into the confirmatory (primary-method) and exploratory families.

    Each comparison is a dict with at least ``method``, ``gate`` ("discrimination"
    or "calibration"), ``severity``, and ``gate_passed``. The confirmatory family
    is every primary-method x gate x severity x tail-assignment comparison within
    one dataset-regime (severities are not separate families); everything else,
    including gated-out primary-method comparisons, is exploratory or "not tested".
    """
    confirmatory = [c for c in comparisons if c.get("method") in PRIMARY_METHODS]
    exploratory = [c for c in comparisons if c.get("method") not in PRIMARY_METHODS]
    return confirmatory, exploratory


def _annotate_confirmatory(
    confirmatory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Holm-adjust the confirmatory family's testable comparisons; mark the rest not tested."""
    testable = [
        c for c in confirmatory if c.get("gate_passed") and c.get("p_value") is not None
    ]
    adjusted = holm_adjust_pvalues([float(c["p_value"]) for c in testable])
    tested_ids = {id(c) for c in testable}
    out = [
        {**c, "adjusted_p_value": p_adj, "status": "tested", "family": "confirmatory"}
        for c, p_adj in zip(testable, adjusted, strict=True)
    ]
    out += [
        {
            **c,
            "adjusted_p_value": None,
            "status": "not tested",
            "family": "confirmatory",
        }
        for c in confirmatory
        if id(c) not in tested_ids
    ]
    return out


def _annotate_exploratory(exploratory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exploratory comparisons keep effects and CIs only; they are never hypothesis-tested.

    Setup §3.6 limits hypothesis tests to the four primary methods. Exploratory
    comparisons therefore carry no p-value and are labelled ``"exploratory"``
    rather than ``"tested"``, even when they pass a deficit gate.
    """
    return [
        {
            **c,
            "p_value": None,
            "adjusted_p_value": None,
            "status": "exploratory",
            "family": "exploratory",
        }
        for c in exploratory
    ]


def apply_holm(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adjust p-values within the confirmatory family; mark gated-out cells "not tested".

    Only gate-passing comparisons carry a testable p-value; Holm adjusts
    across exactly those, and every gated-out primary-method comparison is
    recorded with ``adjusted_p_value=None`` and ``status="not tested"``.
    """
    confirmatory, exploratory = confirmatory_family(comparisons)
    return _annotate_confirmatory(confirmatory) + _annotate_exploratory(exploratory)
