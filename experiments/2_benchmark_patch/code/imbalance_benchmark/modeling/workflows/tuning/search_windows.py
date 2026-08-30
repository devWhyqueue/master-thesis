from __future__ import annotations

from typing import Any

__all__ = [
    "LR_ENVELOPE",
    "STRENGTH_ENVELOPES",
    "CE_ANCHORED_METHODS",
    "initial_window",
    "winner_is_interior",
    "shift_window",
    "expand_grid",
    "new_candidates",
]

# The prior run selected both outer values, so both are interior to this
# rerun's envelope. The active four-point window may still shift outward.
LR_ENVELOPE: list[float] = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

# Audit found these two controls unbounded above (no natural domain ceiling,
# unlike balanced-sampling/weighted-CE/OKO strength 1). Each envelope's
# active four-point window starts at GRIDS[method] (modeling.context) and
# may shift upward one position at a time. CE (that method's parameter=0)
# is a free anchor point below the window for focal only: gamma=0 degenerates
# exactly to plain CE, so its metrics are aliased from CE's already-computed
# candidates rather than trained (see tuning_reduction). ce_soft_f1/mcc train
# under a forced balanced sampler (loaders.FIXED_BALANCED_SAMPLER_METHODS),
# so their weight=0 point is balanced-sampling CE, not CE - it is trained for
# real, not aliased, hence their absence from CE_ANCHORED_METHODS below.
STRENGTH_ENVELOPES: dict[str, list[float]] = {
    "focal": [0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0],
    "ce_soft_f1": [0.0, 0.25, 1.0, 4.0, 16.0, 64.0],
    "ce_soft_mcc": [0.0, 0.25, 1.0, 4.0, 16.0, 64.0],
}

CE_ANCHORED_METHODS = frozenset({"weighted_ce", "balanced_sampling", "focal"})


def initial_window(regime: str, envelope: list[float]) -> list[float]:
    """Return one of three initial four-point windows into a frozen envelope.

    ``regime`` is "low", "current", or "high": low-centered starts at the
    envelope's first value, current-centered reproduces today's frozen
    default window, and high-centered ends at the envelope's last value.
    """
    starts = {"low": 0, "current": len(envelope) // 2 - 1, "high": len(envelope) - 4}
    if regime not in starts:
        raise ValueError(f"Unknown window regime: {regime}")
    start = starts[regime]
    return envelope[start : start + 4]


def winner_is_interior(window: list[float], winner: float) -> bool:
    """A winner is interior when it is neither of the active window's two edges."""
    return winner not in (window[0], window[-1])


def shift_window(
    window: list[float], winner: float, envelope: list[float]
) -> list[float] | None:
    """Slide the active window one position toward an edge winner.

    Reuses the three overlapping envelope values and adds exactly one new
    column. Returns ``None`` when the window already touches the envelope's
    boundary on the winning side, i.e. the search is tuning-limited.
    """
    if winner_is_interior(window, winner):
        raise ValueError("shift_window requires an edge winner")
    start = envelope.index(window[0])
    if winner == window[-1]:
        if start + len(window) >= len(envelope):
            return None
        return envelope[start + 1 : start + 1 + len(window)]
    if start == 0:
        return None
    return envelope[start - 1 : start - 1 + len(window)]


def expand_grid(
    lr_window: list[float], strength_window: list[float] | None
) -> list[dict[str, Any]]:
    """Cross one lr/strength window pair into its full candidate list."""
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
    previous = expand_grid(prev_lr_window, prev_strength_window)
    current = expand_grid(lr_now, strength_now)
    return [cfg for cfg in current if cfg not in previous]
