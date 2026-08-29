from __future__ import annotations

import numpy as np
import pytest

from imbalance_benchmark.analysis.predictors.signals.signal_profile import (
    _nominal_shortage,
)


def test_nominal_shortage_is_weighted_by_difficulty_so_permuted_assignments_differ():
    """Same allocated-count multiset, permuted onto classes of different difficulty,
    must not produce the same score -- an unweighted mean cannot see the assignment.

    Regression for the published bug: BRACS moderate read the same nominal
    shortage for the native, aligned, and reversed tail assignments.
    """
    balanced = {"allocated_counts": {"A": 10, "B": 10, "C": 10}}
    difficulty = {"A": 0.9, "B": 0.1, "C": 0.5}
    hard_class_cut_more = {"allocated_counts": {"A": 2, "B": 5, "C": 10}}
    hard_class_cut_less = {"allocated_counts": {"A": 5, "B": 2, "C": 10}}

    cut_more = _nominal_shortage(balanced, hard_class_cut_more, difficulty)
    cut_less = _nominal_shortage(balanced, hard_class_cut_less, difficulty)

    assert cut_more == pytest.approx(0.9 * np.log(5) + 0.1 * np.log(2))
    assert cut_less == pytest.approx(0.1 * np.log(5) + 0.9 * np.log(2))
    assert cut_more != pytest.approx(cut_less)


def test_nominal_shortage_falls_back_to_unweighted_mean_when_weights_are_zero():
    balanced = {"allocated_counts": {"A": 10, "B": 10}}
    imbalanced = {"allocated_counts": {"A": 5, "B": 5}}

    shortage = _nominal_shortage(balanced, imbalanced, {"A": 0.0, "B": 0.0})

    assert shortage == pytest.approx(np.log(2.0))
