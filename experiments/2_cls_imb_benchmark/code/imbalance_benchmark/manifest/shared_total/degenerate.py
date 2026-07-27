from __future__ import annotations

from typing import Any


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
    """
    balanced_counts = meta["conditions"]["balanced"]["allocated_counts"]
    for assignment, conditions in meta["assignment_conditions"].items():
        for name, condition in conditions.items():
            degenerate = (
                abs(condition["achieved_rho"] - 1.0) <= tolerance
                or condition["allocated_counts"] == balanced_counts
            )
            if degenerate:
                raise ValueError(_degenerate_message(assignment, name, condition))
