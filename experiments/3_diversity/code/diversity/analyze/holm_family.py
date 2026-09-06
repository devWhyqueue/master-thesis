"""The flat, per-dataset confirmatory family that Holm adjustment runs over."""

from __future__ import annotations

from typing import Any

__all__ = ["confirmatory_pvalues"]


def confirmatory_pvalues(
    damage: list[dict[str, Any]],
    interaction: list[dict[str, Any]],
    matched: list[dict[str, Any]],
) -> list[tuple[str, float]]:
    """One flat, per-dataset family: report Sec. "Analysis" adjusts it jointly, not per gate.

    4 damage tests (2 allocations x 2 endpoints, gate-open only) + 2
    interaction tests (always tested) + the matched-vs-unmatched contrast in
    each opened cell.
    """
    entries = [
        (f"damage:{d['allocation']}:{d['endpoint']}", d["p_value"])
        for d in damage
        if d.get("gate_passed") and d.get("p_value") is not None
    ]
    entries += [
        (f"interaction:{i['endpoint']}", i["p_value"])
        for i in interaction
        if i.get("p_value") is not None
    ]
    entries += [
        (f"matched_vs_unmatched:{m['allocation']}:{m['endpoint']}", m["p_value"])
        for m in matched
    ]
    return entries
