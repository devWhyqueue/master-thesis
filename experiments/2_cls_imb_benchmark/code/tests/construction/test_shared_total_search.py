from __future__ import annotations

import random

import pandas as pd
import pytest

from imbalance_benchmark.construction import allocate_counts, effective_rho, max_shared_total
from imbalance_benchmark.manifest.construction_helpers import class_support_counts
from imbalance_benchmark.manifest.shared_total import search as shared_total_search
from imbalance_benchmark.manifest.shared_total.search import (
    _FeasibilityContext,
    _build_search_context,
    _largest_jointly_optimal_total,
    _scan_candidates,
    _Candidate,
    cap_feasible_shared_total,
)
from imbalance_benchmark.manifest.shared_total.severity import severity_aware_upper_bound
from imbalance_benchmark.manifest.statistics import achieved_rho
from imbalance_benchmark.manifest.statistics.selection_capacity import (
    feasible_selection_counts,
)


def _flat_frame(counts: dict[str, int], is_mil: bool) -> pd.DataFrame:
    """One row per unit; MIL treats each row as its own patient/slide."""
    rows = [
        {
            "case_id": f"{name}_{index}",
            "slide_id": f"{name}_{index}",
            "patch_id": f"{name}_{index}_p",
            "cancer_type": name,
            "split": "train",
        }
        for name, count in counts.items()
        for index in range(count)
    ]
    return pd.DataFrame(rows)


# --- brute-force oracle ---------------------------------------------------


def _oracle_candidates(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    assignments: dict[str, list[str]],
) -> list[_Candidate]:
    """Score every total with the slow, obviously-correct effective_rho path.

    Independent of the fast ``_weighted_counts`` bisection this refactor
    introduces: every count here comes from the pre-existing, already-tested
    ``effective_rho`` root-finder feeding ``allocate_counts``, exactly what
    ``_build_conditions`` uses for the real, authoritative construction.
    """
    supports = class_support_counts(train_df, is_mil)
    floor = len(classes) * min_support
    ceiling = max(
        max_shared_total([supports[name] for name in classes], min_support),
        severity_aware_upper_bound(supports, assignments, min_support),
    )
    feasible_counts = feasible_selection_counts(train_df, min_support, is_mil)
    candidates: list[_Candidate] = []
    for total in range(floor, ceiling + 1):
        try:
            balanced_available = [supports[name] for name in classes]
            balanced_rho = effective_rho(balanced_available, 1.0, min_support, total)
            balanced_counts = dict(
                zip(
                    classes,
                    allocate_counts(balanced_available, total, balanced_rho, min_support),
                    strict=True,
                )
            )
        except ValueError:
            continue
        if any(balanced_counts[name] not in feasible_counts[name] for name in classes):
            continue
        worst_moderate = float("inf")
        worst_severe = float("inf")
        feasible = True
        for order in assignments.values():
            available = [supports[name] for name in order]
            try:
                moderate_rho = effective_rho(available, 10.0, min_support, total)
                moderate = dict(
                    zip(order, allocate_counts(available, total, moderate_rho, min_support), strict=True)
                )
                severe_rho = effective_rho(available, 100.0, min_support, total)
                severe = dict(
                    zip(order, allocate_counts(available, total, severe_rho, min_support), strict=True)
                )
            except ValueError:
                feasible = False
                break
            if any(moderate[name] not in feasible_counts[name] for name in order) or any(
                severe[name] not in feasible_counts[name] for name in order
            ):
                feasible = False
                break
            worst_moderate = min(worst_moderate, achieved_rho(moderate))
            worst_severe = min(worst_severe, achieved_rho(severe))
        if not feasible:
            continue
        candidates.append(_Candidate(total, worst_moderate, worst_severe))
    return candidates


def _oracle_cap_feasible_shared_total(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    seed: int,
    independent_floor: int = 10,
    assignments: dict[str, list[str]] | None = None,
) -> int | None:
    """Reference implementation of the whole search using only the slow path.

    Reuses ``_largest_jointly_optimal_total`` (the joint-maxima selection and
    real fixed-pool probe) unchanged, so this isolates the comparison to
    whether the fast per-total scoring agrees with the slow one.
    """
    locked = assignments or {"native": classes}
    ctx = _FeasibilityContext(
        train_df,
        is_mil,
        seed,
        independent_floor,
        min_support,
        locked,
        class_support_counts(train_df, is_mil),
        feasible_selection_counts(train_df, min_support, is_mil),
        classes,
        sorted(
            classes,
            key=lambda name: len(feasible_selection_counts(train_df, min_support, is_mil)[name]),
        ),
    )
    candidates = _oracle_candidates(train_df, classes, min_support, is_mil, locked)
    if not candidates:
        return None
    try:
        return _largest_jointly_optimal_total(ctx, candidates)
    except ValueError:
        return None


def _fast_result(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    seed: int,
    independent_floor: int = 10,
    assignments: dict[str, list[str]] | None = None,
) -> int | None:
    try:
        return cap_feasible_shared_total(
            train_df, classes, min_support, is_mil, seed, independent_floor, assignments
        )
    except ValueError:
        return None


@pytest.mark.parametrize("trial", range(12))
def test_exact_search_matches_the_slow_effective_rho_oracle(trial: int) -> None:
    """The fast bisection-based scan must pick the same total as the slow oracle.

    Randomized small inventories exercise the water-filling shift search
    (``_weighted_counts``) against every rounding/clip edge it can hit,
    without the O(total) cost of running the slow oracle over a realistic
    47,311-total range.
    """
    rng = random.Random(trial)
    min_support = rng.choice([5, 10])
    k = rng.choice([2, 3])
    names = ["A", "B", "C"][:k]
    counts = {name: rng.randint(min_support, min_support * 12) for name in names}
    is_mil = rng.choice([True, False])
    train_df = _flat_frame(counts, is_mil)
    assignments = {"native": names} if k == 2 else {"native": names, "reversed": list(reversed(names))}

    fast = _fast_result(train_df, names, min_support, is_mil, seed=trial, assignments=assignments)
    oracle = _oracle_cap_feasible_shared_total(
        train_df, names, min_support, is_mil, seed=trial, assignments=assignments
    )

    assert fast == oracle


def test_disconnected_feasible_totals_still_finds_the_largest_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contribution cap can accept a smaller total after rejecting a larger one.

    The largest joint-maxima tier for this fixture is the pair of totals
    {55, 56}; blocking both at the pool probe forces the search past that
    entire tier into a *disconnected*, strictly lower tier ({54}) whose
    achieved ratios are not a tie-break within the same maxima but a freshly
    recomputed, smaller pair - exactly the "remove it and recompute the
    maxima exactly" step the search must perform rather than stopping at
    the first failure.
    """
    counts = {"A": 300, "B": 150, "C": 18}
    train_df = _flat_frame(counts, is_mil=True)
    min_support = 10

    ctx, floor, ceiling = _build_search_context(
        train_df, ["A", "B", "C"], min_support, True, 1, 10, None
    )
    candidates = _scan_candidates(ctx, floor, ceiling)
    best_moderate = max(c.worst_moderate for c in candidates)
    best_severe = max(c.worst_severe for c in candidates)
    top_tier = {c.total for c in candidates if c.worst_moderate == best_moderate and c.worst_severe == best_severe}
    assert top_tier == {55, 56}

    real_probe = shared_total_search._total_cap_feasible

    def blocked_probe(ctx_arg, total):
        if total in top_tier:
            return False
        return real_probe(ctx_arg, total)

    monkeypatch.setattr(shared_total_search, "_total_cap_feasible", blocked_probe)

    chosen = _largest_jointly_optimal_total(ctx, candidates)

    assert chosen == 54
    fallback = next(c for c in candidates if c.total == 54)
    assert (fallback.worst_moderate, fallback.worst_severe) != (best_moderate, best_severe)


def test_exact_search_finds_a_narrow_optimum_that_geometric_sampling_could_skip() -> None:
    """A single-total-wide optimum must not be missed by an exhaustive integer scan.

    The old geometric descent sampled totals at exponentially widening
    strides; a peak occupying only one or two integers out of a wide range
    could fall entirely between two sampled points. The exact scan checks
    every integer, so it cannot step over a narrow optimum by construction -
    this pins that guarantee against regression.
    """
    counts = {"A": 300, "B": 150, "C": 21}
    train_df = _flat_frame(counts, is_mil=True)
    min_support = 10

    total = cap_feasible_shared_total(train_df, ["A", "B", "C"], min_support, True, 1)

    ctx, floor, ceiling = _build_search_context(
        train_df, ["A", "B", "C"], min_support, True, 1, 10, None
    )
    candidates = _scan_candidates(ctx, floor, ceiling)
    best_moderate = max(c.worst_moderate for c in candidates)
    best_severe = max(c.worst_severe for c in candidates)
    chosen = next(c for c in candidates if c.total == total)
    assert chosen.worst_moderate == best_moderate
    assert chosen.worst_severe == best_severe


def test_no_simultaneous_maxima_raises_rather_than_a_summed_compromise() -> None:
    """Disjoint moderate/severe peak windows must fail explicitly, not compromise.

    The largest total maximizing the *sum* of moderate and severe achieved
    ratios (the old behavior) is a real, silently-computable number here;
    the exact protocol must refuse it instead of returning it.
    """
    counts = {"A": 400, "B": 400}
    train_df = _flat_frame(counts, is_mil=True)

    with pytest.raises(ValueError, match="simultaneously"):
        cap_feasible_shared_total(train_df, ["A", "B"], 10, True, 1)


def test_absence_of_any_feasible_total_raises() -> None:
    """A class entirely concentrated in one patient/slide is never cap-selectable.

    Every one of B's 15 patches comes from a single case and slide, so no
    count for B - not even at the support floor - fits within the 10%
    per-unit / 5% per-slide contribution caps. No shared total can exist.
    """
    rows = [
        {
            "case_id": f"A_{index}",
            "slide_id": f"A_{index}",
            "patch_id": f"A_{index}_p",
            "cancer_type": "A",
            "split": "train",
        }
        for index in range(100)
    ] + [
        {
            "case_id": "B_0",
            "slide_id": "B_0",
            "patch_id": f"B_0_p{index}",
            "cancer_type": "B",
            "split": "train",
        }
        for index in range(15)
    ]
    train_df = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="independent-support and contribution caps"):
        cap_feasible_shared_total(train_df, ["A", "B"], 10, False, 1)


def test_failed_top_pool_candidate_is_excluded_and_maxima_are_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the largest jointly-optimal total fails the pool probe, retry lower.

    Forcing the single largest joint-maxima total to fail must not surface
    as an overall failure: a smaller total tying the same maxima (or, once
    that tier is exhausted, a new, recomputed pair of maxima) must still be
    found.
    """
    counts = {"A": 300, "B": 150, "C": 21}
    train_df = _flat_frame(counts, is_mil=True)
    min_support = 10
    ctx, floor, ceiling = _build_search_context(
        train_df, ["A", "B", "C"], min_support, True, 1, 10, None
    )
    candidates = _scan_candidates(ctx, floor, ceiling)
    best_moderate = max(c.worst_moderate for c in candidates)
    best_severe = max(c.worst_severe for c in candidates)
    joint_totals = sorted(
        c.total for c in candidates if c.worst_moderate == best_moderate and c.worst_severe == best_severe
    )
    largest_joint_total = joint_totals[-1]

    real_probe = shared_total_search._total_cap_feasible

    def fail_largest(ctx_arg, total):
        if total == largest_joint_total:
            return False
        return real_probe(ctx_arg, total)

    monkeypatch.setattr(shared_total_search, "_total_cap_feasible", fail_largest)

    chosen = _largest_jointly_optimal_total(ctx, candidates)

    assert chosen != largest_joint_total
    assert chosen == cap_feasible_shared_total(
        train_df, ["A", "B", "C"], min_support, True, 1
    ) or chosen in joint_totals


def test_rare_class_floor_bounds_every_candidate_from_below() -> None:
    """No candidate total may allocate any class below the support floor."""
    counts = {"A": 500, "B": 15}
    train_df = _flat_frame(counts, is_mil=True)
    min_support = 15

    total = cap_feasible_shared_total(train_df, ["A", "B"], min_support, True, 1)

    ctx, floor, ceiling = _build_search_context(train_df, ["A", "B"], min_support, True, 1, 10, None)
    assert total >= floor
    assert floor == 2 * min_support


def test_assignment_order_reversal_is_the_binding_constraint() -> None:
    """The worst-case (min) across assignments, not the native order alone, governs."""
    counts = {"A": 500, "B": 60, "C": 60}
    train_df = _flat_frame(counts, is_mil=True)
    min_support = 15
    assignments = {"native": ["A", "B", "C"], "reversed": ["C", "B", "A"]}

    total = cap_feasible_shared_total(
        train_df, ["A", "B", "C"], min_support, True, 1, assignments=assignments
    )

    ctx, floor, ceiling = _build_search_context(
        train_df, ["A", "B", "C"], min_support, True, 1, 10, assignments
    )
    candidates = _scan_candidates(ctx, floor, ceiling)
    chosen = next(c for c in candidates if c.total == total)
    # The reversed order's head class (C, only 60 available) caps every
    # candidate's achieved ratio well below what the native order alone
    # (headed by A, 500 available) could otherwise reach.
    assert chosen.worst_severe <= 60 / min_support


def test_integer_rounding_produces_fractional_achieved_ratios() -> None:
    """Achieved ratio is read from the realized integer counts, not the raw request.

    ``allocate_counts`` rounds and floor-clips real-valued targets to
    integers; the resulting achieved ratio (max count / min count) is
    generally a non-integer fraction of two small counts rather than a
    clean copy of the requested rho. A search that instead reported the
    requested rho, or some other rounded/clamped value, would not show this.
    """
    counts = {"A": 300, "B": 150, "C": 18}
    train_df = _flat_frame(counts, is_mil=True)
    min_support = 10

    ctx, floor, ceiling = _build_search_context(train_df, ["A", "B", "C"], min_support, True, 1, 10, None)
    candidates = _scan_candidates(ctx, floor, ceiling)

    assert any(c.worst_moderate != int(c.worst_moderate) for c in candidates)
    assert any(c.worst_severe != int(c.worst_severe) for c in candidates)


def test_degenerate_totals_at_the_floor_score_as_balanced() -> None:
    """At the exact floor, every class is pinned to min_support: rho collapses to 1."""
    counts = {"A": 500, "B": 200}
    train_df = _flat_frame(counts, is_mil=True)
    min_support = 20

    ctx, floor, ceiling = _build_search_context(train_df, ["A", "B"], min_support, True, 1, 10, None)
    candidates = _scan_candidates(ctx, floor, ceiling)
    floor_candidate = next(c for c in candidates if c.total == floor)

    assert floor_candidate.worst_moderate == 1.0
    assert floor_candidate.worst_severe == 1.0
