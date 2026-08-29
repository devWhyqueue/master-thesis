from __future__ import annotations

from typing import Any

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

    ``balanced_narrow`` is excluded by design: its nominal rho is 1.0 by
    construction (plans/04-crossed-condition-family.md), and only its
    independent-support axis moves - ``reject_degenerate_narrowing`` checks
    that axis instead.
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


def reject_degenerate_narrowing(meta: dict[str, Any], tolerance: float = 0.22) -> None:
    """Refuse a freeze whose narrowed cell never actually shrank its pool.

    ``tolerance=0.22`` is measured (plans/03-independent-support-feasibility.md):
    it sits below half the smallest observed distance from a permitted
    narrowing ratio to 1.0 (0.44565), so every genuinely designated narrow
    arm passes while a near-unchanged one fails.
    """
    for assignment, conditions in meta["assignment_conditions"].items():
        for name, condition in conditions.items():
            ratio = condition.get("narrowed_ratio")
            if not ratio:
                continue
            for class_name, achieved in ratio.items():
                if abs(achieved - 1.0) <= tolerance:
                    raise ValueError(
                        f"Degenerate narrowing {assignment}/{name}/{class_name}: "
                        f"achieved patient ratio={achieved} is within tolerance "
                        "of 1.0 (unnarrowed)"
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


def _reject_constant_independent_shortage(
    units: list[dict[str, Any]], balanced: dict[str, Any], is_mil: bool
) -> None:
    """Only applies once a narrowed condition is present (plan 03's measured
    decision: TCGA-UT/PANDA carry the nominal arm only, and are *expected* to
    show zero independent shortage everywhere)."""
    if not any(u.get("narrowed_ratio") for u in units):
        return
    shortages = {round(_independent_shortage(balanced, u, is_mil), 6) for u in units}
    if len(shortages) <= 1:
        raise ValueError(
            "Degenerate freeze: independent_shortage never varies across comparison "
            "units despite a narrowed condition - defect A reproduced one level down "
            "(plans/04-crossed-condition-family.md)"
        )


def reject_degenerate_freeze(meta: dict[str, Any], is_mil: bool) -> None:
    """Run every freeze-time degeneracy guard (plans/03,04) in one call."""
    reject_degenerate_conditions(meta)
    reject_degenerate_narrowing(meta)
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
    _reject_constant_independent_shortage(units, meta["conditions"]["balanced"], is_mil)
