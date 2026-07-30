from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imbalance_benchmark.modeling.workflows.tuning.search_windows import (
    LR_ENVELOPE,
    STRENGTH_ENVELOPES,
    shift_window,
    winner_is_interior,
)

__all__ = [
    "AxisState",
    "RoundState",
    "decide_next_round",
    "round_payload",
    "all_resolved",
    "new_candidates",
]


@dataclass(frozen=True)
class AxisState:
    """One search axis's active window and resolution status after one round."""

    window: list[float]
    resolved: bool
    tuning_limited: bool


@dataclass(frozen=True)
class RoundState:
    """One method's round-decision outcome: resolve, shift, or mark tuning-limited."""

    method: str
    lr: AxisState
    strength: AxisState | None

    @property
    def resolved(self) -> bool:
        """True once every active axis has an interior winner."""
        return self.lr.resolved and (self.strength is None or self.strength.resolved)

    @property
    def tuning_limited(self) -> bool:
        """True once any axis's envelope is exhausted at an edge winner."""
        return self.lr.tuning_limited or (
            self.strength is not None and self.strength.tuning_limited
        )

    @property
    def next_lr_window(self) -> list[float] | None:
        """The window to evaluate next round, or ``None`` if nothing more is needed."""
        return None if self.lr.resolved or self.lr.tuning_limited else self.lr.window

    @property
    def next_strength_window(self) -> list[float] | None:
        """The strength window to evaluate next round, or ``None`` if none is needed."""
        if (
            self.strength is None
            or self.strength.resolved
            or self.strength.tuning_limited
        ):
            return None
        return self.strength.window


def _axis_state(window: list[float], winner: float, envelope: list[float]) -> AxisState:
    """Resolve one axis: interior winner resolves it, an edge winner shifts or limits it."""
    if winner_is_interior(window, winner):
        return AxisState(window, resolved=True, tuning_limited=False)
    shifted = shift_window(window, winner, envelope)
    if shifted is None:
        return AxisState(window, resolved=False, tuning_limited=True)
    return AxisState(shifted, resolved=False, tuning_limited=False)


def decide_next_round(
    method: str,
    winning_config: dict[str, Any],
    lr_window: list[float],
    strength_window: list[float] | None = None,
) -> RoundState:
    """Decide one method's next tuning round from its winning configuration.

    Each active axis (learning rate, and method-specific strength for the
    audited unbounded controls) is judged independently: an interior winner
    resolves that axis, an edge winner shifts the window one position and
    reuses the three overlapping values, and an edge winner against an
    envelope already at that boundary marks the axis tuning-limited.
    """
    lr_state = (
        _axis_state(lr_window, winning_config["lr"], LR_ENVELOPE)
        if "lr" in winning_config
        else AxisState(lr_window, resolved=True, tuning_limited=False)
    )
    strength_state = None
    if strength_window is not None and method in STRENGTH_ENVELOPES:
        strength_state = _axis_state(
            strength_window, winning_config["parameter"], STRENGTH_ENVELOPES[method]
        )
    return RoundState(method, lr_state, strength_state)


def round_payload(states: dict[str, RoundState]) -> dict[str, Any]:
    """Serialize one round's decisions into the signed tuning-round-state shape.

    ``next_*_window`` is ``None`` once an axis is resolved or tuning-limited,
    which is what a decide step reads to know whether another round is owed.
    """
    return {
        method: {
            "resolved": state.resolved,
            "tuning_limited": state.tuning_limited,
            "lr_window": state.lr.window,
            "next_lr_window": state.next_lr_window,
            "strength_window": state.strength.window if state.strength else None,
            "next_strength_window": state.next_strength_window,
        }
        for method, state in states.items()
    }


def all_resolved(states: dict[str, RoundState]) -> bool:
    """True once every method's search is resolved or correctly marked tuning-limited.

    This is the tuning lock: confirmation may only start once it holds for
    every required method of a condition.
    """
    return all(state.resolved or state.tuning_limited for state in states.values())


def _cross(
    lr_window: list[float], strength_window: list[float] | None
) -> list[dict[str, Any]]:
    if strength_window is None:
        return [{"lr": lr} for lr in lr_window]
    return [{"parameter": p, "lr": lr} for p in strength_window for lr in lr_window]


def new_candidates(
    prev_lr_window: list[float],
    next_lr_window: list[float] | None,
    prev_strength_window: list[float] | None = None,
    next_strength_window: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Return exactly the configs in the next round's grid that are not in the current one.

    Preserves the full factorial: when both axes shift in the same round, the
    new corner (new lr, new strength) is included alongside the two new
    edges, since a single outward corner cannot rule out an LR-strength
    interaction.
    """
    lr_now = next_lr_window if next_lr_window is not None else prev_lr_window
    strength_now = (
        next_strength_window
        if next_strength_window is not None
        else prev_strength_window
    )
    previous = _cross(prev_lr_window, prev_strength_window)
    current = _cross(lr_now, strength_now)
    return [cfg for cfg in current if cfg not in previous]
