from __future__ import annotations

from imbalance_benchmark.modeling.context import LEARNING_RATE_GRID
from imbalance_benchmark.modeling.workflows.tuning.search_windows import (
    LR_ENVELOPE,
    STRENGTH_ENVELOPES,
    initial_window,
    new_candidates,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_rounds import (
    all_resolved,
    decide_next_round,
    round_payload,
)


def test_context_default_window_matches_the_envelopes_current_window():
    """Guard the deliberate literal duplication between context and search_windows."""
    assert LEARNING_RATE_GRID == initial_window("current", LR_ENVELOPE)


def test_interior_lr_winner_resolves_the_axis():
    state = decide_next_round("ce", {"lr": LEARNING_RATE_GRID[1]}, LEARNING_RATE_GRID)
    assert state.resolved
    assert not state.tuning_limited
    assert state.next_lr_window is None


def test_edge_lr_winner_shifts_and_is_unresolved():
    state = decide_next_round("ce", {"lr": LEARNING_RATE_GRID[-1]}, LEARNING_RATE_GRID)
    assert not state.resolved
    assert not state.tuning_limited
    assert state.next_lr_window is not None
    assert set(LEARNING_RATE_GRID[1:]) <= set(state.next_lr_window)


def test_lr_envelope_exhaustion_marks_tuning_limited():
    top_window = LR_ENVELOPE[-4:]
    state = decide_next_round("ce", {"lr": top_window[-1]}, top_window)
    assert not state.resolved
    assert state.tuning_limited
    assert state.next_lr_window is None


def test_post_hoc_method_has_no_lr_axis_and_resolves_trivially():
    state = decide_next_round(
        "post_hoc_logit_adjustment", {"parameter": 1.0}, LEARNING_RATE_GRID
    )
    assert state.lr.resolved
    assert state.strength is None
    assert state.resolved


def test_method_without_a_strength_envelope_has_no_strength_axis():
    state = decide_next_round(
        "weighted_ce",
        {"lr": LEARNING_RATE_GRID[1], "parameter": 1.0},
        LEARNING_RATE_GRID,
        strength_window=[0.25, 0.5, 0.75, 1.0],
    )
    assert state.strength is None


def test_focal_strength_edge_winner_shifts_independently_of_lr():
    window = STRENGTH_ENVELOPES["focal"][1:5]
    state = decide_next_round(
        "focal",
        {"lr": LEARNING_RATE_GRID[1], "parameter": window[-1]},
        LEARNING_RATE_GRID,
        strength_window=window,
    )
    assert state.lr.resolved
    assert state.strength is not None
    assert not state.strength.resolved
    assert state.next_strength_window == STRENGTH_ENVELOPES["focal"][2:6]


def test_focal_strength_envelope_exhausts_at_top():
    window = STRENGTH_ENVELOPES["focal"][-4:]
    state = decide_next_round(
        "focal",
        {"lr": LEARNING_RATE_GRID[1], "parameter": window[-1]},
        LEARNING_RATE_GRID,
        strength_window=window,
    )
    assert state.strength.tuning_limited
    assert state.tuning_limited
    assert state.next_strength_window is None


def test_strength_exhaustion_pins_a_still_shifting_lr_to_what_this_round_trained():
    """Regression: found live on the cluster (BRACS severe, ce_soft_mcc).
    Strength hits its envelope ceiling and freezes the whole method while lr
    independently wins an edge case that would normally shift - before this
    fix, the frozen state recorded lr's speculative next window (never
    actually submitted as a round), so final reduction could never find a
    registered candidate for it. lr must instead be pinned to the window it
    was actually evaluated against this round."""
    strength_top = STRENGTH_ENVELOPES["ce_soft_mcc"][-4:]  # already at the ceiling
    lr_window = LEARNING_RATE_GRID
    state = decide_next_round(
        "ce_soft_mcc",
        {"lr": lr_window[0], "parameter": strength_top[-1]},
        lr_window,
        strength_window=strength_top,
    )
    assert state.tuning_limited
    assert state.strength.tuning_limited
    assert state.lr.tuning_limited  # pinned, not left mid-shift
    assert state.lr.window == lr_window  # what this round actually trained
    assert state.next_lr_window is None


def test_resolved_requires_every_active_axis_interior():
    window = STRENGTH_ENVELOPES["ce_soft_f1"][1:5]
    state = decide_next_round(
        "ce_soft_f1",
        {"lr": LEARNING_RATE_GRID[-1], "parameter": window[1]},
        LEARNING_RATE_GRID,
        strength_window=window,
    )
    assert state.strength.resolved
    assert not state.lr.resolved
    assert not state.resolved


def test_round_payload_carries_next_windows_only_while_unresolved():
    resolved = decide_next_round("ce", {"lr": LEARNING_RATE_GRID[1]}, LEARNING_RATE_GRID)
    shifting = decide_next_round("ce", {"lr": LEARNING_RATE_GRID[-1]}, LEARNING_RATE_GRID)
    payload = round_payload({"ce_resolved": resolved, "ce_shifting": shifting})
    assert payload["ce_resolved"]["resolved"] is True
    assert payload["ce_resolved"]["next_lr_window"] is None
    assert payload["ce_shifting"]["resolved"] is False
    assert payload["ce_shifting"]["next_lr_window"] is not None


def test_new_candidates_lr_only_shift_adds_exactly_the_new_column():
    prev = LEARNING_RATE_GRID
    nxt = LR_ENVELOPE[3:7]
    added = new_candidates(prev, nxt)
    assert added == [{"lr": LR_ENVELOPE[6]}]


def test_new_candidates_lr_shift_crossed_with_a_fixed_strength_grid():
    prev_lr, next_lr = LEARNING_RATE_GRID, LR_ENVELOPE[3:7]
    strength = [0.25, 0.5, 0.75, 1.0]
    added = new_candidates(prev_lr, next_lr, strength, strength)
    assert added == [{"parameter": p, "lr": LR_ENVELOPE[6]} for p in strength]


def test_new_candidates_strength_only_shift_adds_exactly_the_new_row():
    lr = LEARNING_RATE_GRID
    prev_strength = STRENGTH_ENVELOPES["focal"][1:5]
    next_strength = STRENGTH_ENVELOPES["focal"][2:6]
    added = new_candidates(lr, None, prev_strength, next_strength)
    assert added == [{"parameter": next_strength[-1], "lr": v} for v in lr]


def test_new_candidates_both_axes_shift_includes_the_new_corner():
    prev_lr, next_lr = LEARNING_RATE_GRID, LR_ENVELOPE[3:7]
    prev_strength = STRENGTH_ENVELOPES["focal"][1:5]
    next_strength = STRENGTH_ENVELOPES["focal"][2:6]
    added = new_candidates(prev_lr, next_lr, prev_strength, next_strength)
    # next window is 4x4=16; only the 3x3=9 pairs surviving both shifts overlap
    # the previous window, so 16-9=7 pairs are new, including the new corner.
    assert len(added) == 7
    assert {"parameter": next_strength[-1], "lr": next_lr[-1]} in added
    assert len(added) == len({(c["parameter"], c["lr"]) for c in added})


def test_new_candidates_nothing_shifting_is_empty():
    assert new_candidates(LEARNING_RATE_GRID, None) == []


def test_all_resolved_true_only_when_every_method_resolved_or_limited():
    resolved = decide_next_round("ce", {"lr": LEARNING_RATE_GRID[1]}, LEARNING_RATE_GRID)
    top_window = LR_ENVELOPE[-4:]
    limited = decide_next_round("ce", {"lr": top_window[-1]}, top_window)
    shifting = decide_next_round("ce", {"lr": LEARNING_RATE_GRID[-1]}, LEARNING_RATE_GRID)
    assert all_resolved({"a": resolved, "b": limited})
    assert not all_resolved({"a": resolved, "b": shifting})
