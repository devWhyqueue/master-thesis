"""Unit tests for exp-29's Latin-square tail-class assignment (shift semantics and rank derivation)."""

from __future__ import annotations

import numpy as np
import pytest
from imbalance_benchmark.datasets.bracs.metadata import LABELS

from prevalence import G
from prevalence.fit import class_counts, class_permutation

from assignment import ARMS, ATYPICAL, NEW_FIT_ARMS, RATIOS, REUSED_ARMS, SHIFTS
from assignment.fit import shifted_perm
from assignment.rank import _rank_of

N_CLASSES = 7  # BRACS-sized
_NAMES = [f"c{i}" for i in range(N_CLASSES)]
_AVAILABLE = [G * 160] * N_CLASSES
_POOL_COUNTS = [5000, 3000, 2000, 1200, 700, 300, 100]
_PERM = class_permutation(split_idx=0, draw_idx=0, num_classes=N_CLASSES)


def _named_counts(perm: np.ndarray, arm: str = "r10") -> dict[str, int]:
    counts = class_counts(arm, perm, _AVAILABLE, _POOL_COUNTS, G)
    return {_NAMES[i]: counts[i] for i in range(N_CLASSES)}


def test_shift_zero_equals_class_permutation() -> None:
    """Shift k = 0 is exp-26's own permutation, unmodified (the basis for arm reuse)."""
    np.testing.assert_array_equal(shifted_perm(_PERM, 0), _PERM)


def test_each_class_takes_each_rank_once_across_shifts() -> None:
    """Latin square: over the 7 cyclic shifts, every class visits every rank exactly once."""
    ranks_seen: dict[int, set[int]] = {c: set() for c in range(N_CLASSES)}
    for k in range(N_CLASSES):
        for rank_pos, cls_idx in enumerate(shifted_perm(_PERM, k)):
            ranks_seen[int(cls_idx)].add(rank_pos)
    for c in range(N_CLASSES):
        assert ranks_seen[c] == set(range(N_CLASSES))


def test_count_multiset_identical_across_shifts() -> None:
    """A cyclic shift relabels which class gets which count; the set of counts is unchanged."""
    baseline = sorted(class_counts("r10", _PERM, _AVAILABLE, _POOL_COUNTS, G))
    for k in range(1, N_CLASSES):
        shifted = sorted(class_counts("r10", shifted_perm(_PERM, k), _AVAILABLE, _POOL_COUNTS, G))
        assert shifted == baseline


@pytest.mark.parametrize("k", range(N_CLASSES))
def test_tail_class_equals_argmin_of_counts(k: int) -> None:
    """The class at a shift's last rank position holds that shift's minimum count."""
    shifted = shifted_perm(_PERM, k)
    counts = class_counts("r10", shifted, _AVAILABLE, _POOL_COUNTS, G)
    assert int(shifted[-1]) == int(np.argmin(counts))


@pytest.mark.parametrize("k", range(N_CLASSES))
def test_rank_of_matches_shift_positions(k: int) -> None:
    """Analysis's ``_rank_of`` (derived from stored class_counts) agrees with the fit-side shift."""
    shifted = shifted_perm(_PERM, k)
    ranks = _rank_of(_named_counts(shifted), _NAMES)
    for rank_pos, cls_idx in enumerate(shifted):
        assert ranks[_NAMES[cls_idx]] == rank_pos


def test_arms_cover_every_shift_and_ratio() -> None:
    """NEW_FIT_ARMS is exactly the 6 non-zero shifts x 2 ratios; REUSED_ARMS is r1/r10/r100."""
    assert set(NEW_FIT_ARMS) == {f"a{k}_r{r}" for r in RATIOS for k in SHIFTS}
    assert len(NEW_FIT_ARMS) == 6 * len(RATIOS) == 12
    assert REUSED_ARMS == ("r1",) + tuple(f"r{r}" for r in RATIOS)
    assert set(ARMS) == set(REUSED_ARMS) | set(NEW_FIT_ARMS)


def test_atypical_are_real_bracs_labels() -> None:
    """The pre-specified contrast names two of BRACS's actual 7 subtype labels."""
    assert set(ATYPICAL) <= set(LABELS)
    assert len(ATYPICAL) == 2
