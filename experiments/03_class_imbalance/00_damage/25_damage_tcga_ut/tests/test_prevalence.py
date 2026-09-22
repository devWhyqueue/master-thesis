"""Unit tests for the prevalence-imbalance class-count allocation and per-patient row nesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from centre.cohort import patient_rows

from imbalance_benchmark.manifest.statistics import achieved_rho

from prevalence import ARMS, BALANCED, DEPTH, G, RATIOS
from prevalence.analyze import _native_gap, _slope
from prevalence.fit import class_counts, class_permutation, _patient_counts

N_CLASSES = 7  # BRACS-sized
_AVAILABLE = [G * DEPTH] * N_CLASSES
_POOL_COUNTS = [5000, 3000, 2000, 1200, 700, 300, 100]
_PERM = class_permutation(split_idx=0, draw_idx=0, num_classes=N_CLASSES)
_TOTAL = G * BALANCED * N_CLASSES


@pytest.mark.parametrize("arm", ARMS)
def test_class_counts_sum_to_budget(arm: str) -> None:
    """Every arm's per-class counts sum to T = G x BALANCED x num_classes."""
    counts = class_counts(arm, _PERM, _AVAILABLE, _POOL_COUNTS)
    assert len(counts) == N_CLASSES
    assert sum(counts) == _TOTAL


@pytest.mark.parametrize("arm", ARMS)
def test_tail_count_at_least_one_patch_per_patient(arm: str) -> None:
    """Every class keeps at least one patch per drawn patient (min_support = G)."""
    counts = class_counts(arm, _PERM, _AVAILABLE, _POOL_COUNTS)
    assert min(counts) >= G


def test_r1_is_exactly_balanced() -> None:
    """r1 gives every class the same count: G x BALANCED patches."""
    counts = class_counts("r1", _PERM, _AVAILABLE, _POOL_COUNTS)
    assert counts == [G * BALANCED] * N_CLASSES


@pytest.mark.parametrize("r", RATIOS[1:])
def test_realized_rho_matches_target(r: int) -> None:
    """Realized rho (max/min class count) matches the requested ratio within rounding tolerance."""
    counts = class_counts(f"r{r}", _PERM, _AVAILABLE, _POOL_COUNTS)
    realized = achieved_rho(dict(enumerate(counts)))
    assert realized == pytest.approx(r, rel=0.05)


def test_class_permutation_is_a_bijection_and_deterministic() -> None:
    """class_permutation returns a permutation of range(num_classes), reproducible for the same draw."""
    perm = class_permutation(split_idx=1, draw_idx=2, num_classes=N_CLASSES)
    assert sorted(perm.tolist()) == list(range(N_CLASSES))
    assert class_permutation(split_idx=1, draw_idx=2, num_classes=N_CLASSES).tolist() == perm.tolist()


@pytest.mark.parametrize("total", [640, 641, 645, 20, 3200])
def test_patient_counts_sum_to_class_total(total: int) -> None:
    """Per-patient counts split one class's total exactly across its G patients."""
    per_patient = _patient_counts(total)
    assert len(per_patient) == G
    assert sum(per_patient) == total
    assert max(per_patient) - min(per_patient) <= 1


def _patient_df(case_id: str, n_patches: int = DEPTH) -> pd.DataFrame:
    """One patient's manifest rows, single slide, stable patch order."""
    return pd.DataFrame(
        [
            {
                "cancer_type": "cls",
                "case_id": case_id,
                "slide_id": f"{case_id}_s0",
                "patch_id": f"{case_id}_p{patch:04d}",
            }
            for patch in range(n_patches)
        ]
    )


def test_nested_subset_property_across_arm_ordering() -> None:
    """A smaller per-patient count's rows are a strict, order-preserving prefix of a larger count's rows."""
    class_df = _patient_df("TCGA-01-0001")
    counts = sorted({_patient_counts(class_counts(a, _PERM, _AVAILABLE, _POOL_COUNTS)[0])[0] for a in ARMS})
    rows_by_count = {m: patient_rows(class_df, ["TCGA-01-0001"], m) for m in counts if m > 0}
    ordered = sorted(rows_by_count)
    for lo, hi in zip(ordered, ordered[1:]):
        assert rows_by_count[lo] == rows_by_count[hi][: len(rows_by_count[lo])]


def test_per_patient_split_sums_to_class_total_for_every_arm() -> None:
    """Summing patient_rows lengths across the G patients recovers the class's realized total."""
    for arm in ARMS:
        counts = class_counts(arm, _PERM, _AVAILABLE, _POOL_COUNTS)
        for total in counts:
            per_patient = _patient_counts(total)
            assert sum(per_patient) == total
            assert all(0 <= m <= DEPTH for m in per_patient)


def test_slope_and_native_gap_recover_log_linear_curve() -> None:
    """A curve exactly linear in log2(r) gives its slope, and N on that curve gives zero gap."""
    ba = {f"r{r}": np.full(3, 70.0 - 0.5 * np.log2(r)) for r in RATIOS}
    rho = {f"r{r}": float(r) for r in RATIOS} | {"N": 20.0}
    ba["N"] = np.full(3, float(np.interp(20.0, RATIOS, [b[0] for b in ba.values()])))
    np.testing.assert_allclose(_slope(ba, RATIOS), -0.5)
    np.testing.assert_allclose(_native_gap(ba, rho), 0.0, atol=1e-12)


@pytest.mark.parametrize("rho", RATIOS)
def test_bracs_g10_grid_is_feasible(rho: int) -> None:
    """G = 10 (BRACS override) reaches every ratio at the exact budget without clamping the head."""
    g = 10
    counts = class_counts(f"r{rho}", _PERM, [g * DEPTH] * N_CLASSES, _POOL_COUNTS, g)
    assert sum(counts) == g * BALANCED * N_CLASSES
    assert achieved_rho(dict(enumerate(counts))) == pytest.approx(rho, rel=0.05)
