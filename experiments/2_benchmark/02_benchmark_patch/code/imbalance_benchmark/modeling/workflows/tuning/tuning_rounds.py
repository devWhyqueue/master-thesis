from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imbalance_benchmark.modeling.workflows.tuning.search_windows import (
    LR_ENVELOPE,
    STRENGTH_ENVELOPES,
    shift_window,
    winner_is_interior,
)
from imbalance_benchmark.modeling.workflows.tuning.aggregation.candidate_registry import (
    load_registry,
    register_candidates,
    registry_lookup,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import ShardSpec

__all__ = [
    "AxisState",
    "RoundState",
    "decide_next_round",
    "round_payload",
    "all_resolved",
    "new_configs_for_round",
    "resolve_round_specs",
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


def _freeze_orphaned_axes(
    state: RoundState, lr_window: list[float], strength_window: list[float] | None
) -> RoundState:
    """Pin an axis mid-shift to what this round actually trained.

    Either axis's independent envelope exhaustion freezes the whole method
    (report protocol: search stops once any control is tuning-limited), but
    the other axis may have computed a real shift this same round. That
    shifted window is never submitted as a next round, so it must not be
    recorded as this axis's state either - only the window this round
    actually trained is real.
    """
    lr = state.lr
    if not (lr.resolved or lr.tuning_limited):
        lr = AxisState(lr_window, resolved=False, tuning_limited=True)
    strength = state.strength
    if (
        strength is not None
        and strength_window is not None
        and not (strength.resolved or strength.tuning_limited)
    ):
        strength = AxisState(strength_window, resolved=False, tuning_limited=True)
    return RoundState(state.method, lr, strength)


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
    state = RoundState(method, lr_state, strength_state)
    if state.tuning_limited:
        return _freeze_orphaned_axes(state, lr_window, strength_window)
    return state


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


def _round_zero_specs(
    root: Path,
    condition: str,
    phase: str,
    method: str,
    active_grid: list[dict[str, Any]],
) -> list[ShardSpec]:
    """Address the frozen initial window exactly as before and register it."""
    specs = [
        ShardSpec(condition, method, index, phase) for index in range(len(active_grid))
    ]
    register_candidates(root, condition, method, active_grid, round_index=0)
    return specs


def new_configs_for_round(
    root: Path, condition: str, method: str, active_grid: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return this round's active-grid configs not already trained in an earlier round.

    Pure lookup (no registration), so a shard task can independently derive
    exactly which of its round's configs it must actually train, matching
    what the later reduce step will independently register.
    """
    registry = load_registry(root, condition)
    return [
        config
        for config in active_grid
        if registry_lookup(registry, method, config) is None
    ]


def _later_round_specs(
    root: Path,
    condition: str,
    phase: str,
    method: str,
    active_grid: list[dict[str, Any]],
    round_index: int,
) -> list[ShardSpec]:
    """Reuse each already-trained value from wherever it lives; register the rest."""
    new_configs = new_configs_for_round(root, condition, method, active_grid)
    if new_configs:
        register_candidates(root, condition, method, new_configs, round_index)
    registry = load_registry(root, condition)
    specs = []
    for config in active_grid:
        found = registry_lookup(registry, method, config)
        if found is None:
            raise RuntimeError(f"Candidate not registered after this round: {config}")
        source_round, index = found
        specs.append(ShardSpec(condition, method, index, phase, round=source_round))
    return specs


def resolve_round_specs(
    root: Path,
    condition: str,
    phase: str,
    method: str,
    active_grid: list[dict[str, Any]],
    round_index: int,
) -> list[ShardSpec]:
    """Resolve one round's active grid to wherever each candidate was trained.

    Round 0 is the frozen initial window, addressed exactly as before. A
    later round's grid may reuse values a previous round already trained;
    those are looked up in the cross-round registry, and only genuinely
    new values are registered against this round.
    """
    if round_index == 0:
        return _round_zero_specs(root, condition, phase, method, active_grid)
    return _later_round_specs(root, condition, phase, method, active_grid, round_index)
