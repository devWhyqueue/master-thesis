from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.predictors.rq3_features import _independent_shortage


def _degenerate_message(assignment: str, name: str, condition: dict[str, Any]) -> str:
    return (
        f"Degenerate {assignment}/{name} condition: "
        f"achieved_rho={condition['achieved_rho']} "
        f"(requested {condition['requested_rho']}), "
        f"limiting_class={condition['limiting_class']}, "
        "binding_independent_support_constraint="
        f"{condition['binding_independent_support_constraint']}"
    )


def reject_degenerate_conditions(meta: dict[str, Any], tolerance: float = 1e-6) -> None:
    """Refuse a freeze whose moderate/severe conditions never left balanced.

    A degenerate construction (``achieved_rho`` within ``tolerance`` of 1.0,
    or counts identical to the balanced manifest) means the controlled
    comparison did not happen, and every downstream gate would divide by a
    near-zero denominator. Moderate is not compared against severe directly:
    an adversarial assignment (e.g. "reversed") can legitimately tie both at
    a real head-capacity ceiling without collapsing to balanced.

    Rho-one spread conditions are excluded by design; their independent-support
    axis is checked separately.
    """
    balanced_counts = meta["conditions"]["balanced"]["allocated_counts"]
    for assignment, conditions in meta["assignment_conditions"].items():
        for name, condition in conditions.items():
            if condition.get("requested_rho") == 1.0:
                continue
            degenerate = (
                abs(condition["achieved_rho"] - 1.0) <= tolerance
                or condition["allocated_counts"] == balanced_counts
            )
            if degenerate:
                raise ValueError(_degenerate_message(assignment, name, condition))


def reject_degenerate_spreading(meta: dict[str, Any], floor: float = 0.25) -> None:
    """Refuse a spread arm whose mean achieved log-shortage is too small."""
    for assignment, conditions in meta["assignment_conditions"].items():
        for name, condition in conditions.items():
            ratio = condition.get("spread_ratio")
            if not ratio:
                continue
            shortage = float(np.mean([np.log(value) for value in ratio.values()]))
            if shortage < floor:
                raise ValueError(
                    f"Degenerate spreading {assignment}/{name}: "
                    f"mean achieved log-shortage={shortage} is below floor={floor}"
                )


def reject_non_nested_pools(
    concentrated: dict[str, pd.DataFrame], spread: dict[str, pd.DataFrame]
) -> None:
    """Refuse a spread arm that swaps patients instead of adding them."""
    for class_name, pool in concentrated.items():
        if not set(pool["case_id"]).issubset(spread[class_name]["case_id"]):
            raise ValueError(
                f"Non-nested spread pool for {class_name}: concentrated patients lost"
            )


def _comparison_units(meta: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        condition
        for conditions in meta["assignment_conditions"].values()
        for condition in conditions.values()
    ]


def _reject_constant_rho(units: list[dict[str, Any]]) -> None:
    if len({round(float(u["achieved_rho"]), 6) for u in units}) <= 1:
        raise ValueError(
            "Degenerate freeze: achieved_rho never varies across comparison units"
        )


def _reference_for_unit(
    meta: dict[str, Any], assignment: str, name: str
) -> dict[str, Any]:
    if name in {"balanced", "severe_spread"}:
        return meta["assignment_conditions"][assignment]["balanced_spread"]
    return meta["conditions"]["balanced"]


def _reject_constant_independent_shortage(meta: dict[str, Any], is_mil: bool) -> None:
    """Require variation whenever this freeze contains a spread arm."""
    if not any(
        unit.get("spread_ratio")
        for conditions in meta["assignment_conditions"].values()
        for unit in conditions.values()
    ):
        return
    shortages = {
        round(
            _independent_shortage(
                _reference_for_unit(meta, assignment, name), unit, is_mil
            ),
            6,
        )
        for assignment, conditions in meta["assignment_conditions"].items()
        for name, unit in conditions.items()
        if name != "balanced_spread"
    }
    if len(shortages) <= 1:
        raise ValueError(
            "Degenerate freeze: independent_shortage never varies across comparison "
            "units despite a spread condition - defect A reproduced one level down "
            "(plans/04-crossed-condition-family.md)"
        )


def reject_degenerate_freeze(meta: dict[str, Any], is_mil: bool) -> None:
    """Run every freeze-time degeneracy guard (plans/03,04) in one call."""
    reject_degenerate_conditions(meta)
    reject_degenerate_spreading(meta)
    reject_constant_signal_axes(meta, is_mil)


def reject_constant_signal_axes(meta: dict[str, Any], is_mil: bool) -> None:
    """Refuse a freeze where the rho or independent-support axis never varies.

    Cheap, freeze-time proxy for two of the four axes ``signal_profile.json``
    will compute: log(rho) reads straight off ``achieved_rho``; independent
    support reuses the same ``_independent_shortage`` formula RQ3 does. This
    is the guard that would have caught defect A (independent-support
    shortage structurally zero in all 18 comparison units) before spending
    signals/match/tune/confirm compute on it.
    """
    units = _comparison_units(meta)
    if not units:
        return
    _reject_constant_rho(units)
    _reject_constant_independent_shortage(meta, is_mil)
