from __future__ import annotations

from typing import Any

__all__ = ["controlled_assignments_for_condition", "scoped_assignments"]


def controlled_assignments_for_condition(
    freeze: dict[str, Any], condition: str, assignments: tuple[str, ...]
) -> tuple[str, ...]:
    """Assignments where this controlled condition was actually constructed.

    moderate/severe are universal across every assignment on every dataset,
    but balanced_narrow/severe_narrow are not (plans/03-independent-support-
    feasibility.md, plans/04-crossed-condition-family.md): whether a given
    (assignment, condition) pair exists is a fact of this split's own frozen
    ``assignment_conditions``, not something callers may assume. An empty
    result means "skip this condition for this split", not a crash.
    """
    return tuple(
        a
        for a in assignments
        if condition in freeze.get("assignment_conditions", {}).get(a, {})
    )


def scoped_assignments(
    condition: str,
    freeze: dict[str, Any],
    assignments: tuple[str, ...],
    placeholder: str = "native",
) -> tuple[str, ...]:
    """Assignments a tune/confirm selection for this condition applies to.

    natural/balanced are assignment-independent, keyed under ``placeholder``
    (tuning: the real "native" key; confirm: the "unassigned" convention it
    translates back to "native" at lookup time). Every other condition is
    scoped via ``controlled_assignments_for_condition``; an empty result
    means skip, not a crash.
    """
    if condition in {"natural", "balanced"}:
        return (placeholder,)
    return controlled_assignments_for_condition(freeze, condition, assignments)
