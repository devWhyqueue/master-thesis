"""Unit tests for exp-27's prior/support reweighting and post-hoc logit adjustment."""

from __future__ import annotations

import numpy as np
import pytest

from prevalence import BALANCED, DEPTH, G, RATIOS
from prevalence.fit import _prior_weights, class_counts, class_permutation

from cause import ADJUST_SOURCE, FIT_SOURCE, NEW_FIT_ARMS, RATIOS_NEW
from cause.fit import _adjusted_preds

N_CLASSES = 7  # BRACS-sized
_AVAILABLE = [G * DEPTH] * N_CLASSES
_POOL_COUNTS = [5000, 3000, 2000, 1200, 700, 300, 100]
_PERM = class_permutation(split_idx=0, draw_idx=0, num_classes=N_CLASSES)
_TOTAL = G * BALANCED * N_CLASSES


def _counts(arm: str) -> list[int]:
    return class_counts(arm, _PERM, _AVAILABLE, _POOL_COUNTS)


def _class_shares(counts: list[int], weight: np.ndarray) -> np.ndarray:
    """Weighted class shares (sum of weight per class / N), in class order."""
    total = float(sum(counts))
    shares = []
    start = 0
    for n in counts:
        shares.append(float(weight[start : start + n].sum()) / total)
        start += n
    return np.asarray(shares)


def test_weights_sum_to_n() -> None:
    """Sigma w = N for every rho, so the lambda grid keeps fit_multinomial_logistic's meaning."""
    n = _counts("r1")
    for r in RATIOS_NEW:
        w = _prior_weights(n, _counts(f"r{r}"))
        assert w.sum() == pytest.approx(sum(n))


def test_r_arms_weight_one() -> None:
    """A data arm reweighted toward its own class shares gets unit weight everywhere."""
    n = _counts("r10")
    w = _prior_weights(n, n)
    np.testing.assert_allclose(w, 1.0)


@pytest.mark.parametrize("r", RATIOS_NEW)
def test_p_weighted_shares_match_r_rho(r: int) -> None:
    """P{rho}: r1's rows reweighted toward r{rho}'s class shares reproduce r{rho}'s shares."""
    n, t = _counts("r1"), _counts(f"r{r}")
    weighted = _class_shares(n, _prior_weights(n, t))
    target = np.asarray(t, dtype=np.float64) / sum(t)
    np.testing.assert_allclose(weighted, target, atol=1e-9)


@pytest.mark.parametrize("r", RATIOS_NEW)
def test_s_weighted_shares_balanced(r: int) -> None:
    """S{rho}: r{rho}'s rows reweighted toward r1's shares reproduce the balanced 1/K shares."""
    n, t = _counts(f"r{r}"), _counts("r1")
    weighted = _class_shares(n, _prior_weights(n, t))
    np.testing.assert_allclose(weighted, np.full(N_CLASSES, 1.0 / N_CLASSES), atol=1e-9)


def test_new_fit_arms_cover_every_non_unit_ratio() -> None:
    """Exactly P{rho} and S{rho} for every rho in RATIOS (minus the shared r1 baseline)."""
    assert RATIOS_NEW == RATIOS[1:]
    assert (
        set(FIT_SOURCE)
        == set(NEW_FIT_ARMS)
        == {f"{fam}{r}" for fam in ("P", "S") for r in RATIOS_NEW}
    )
    assert set(ADJUST_SOURCE) == {
        f"{fam}{r}" for fam in ("LP", "Lr") for r in RATIOS_NEW
    }


def test_fit_source_sends_p_to_r1_rows_and_s_to_r_rho_rows() -> None:
    """P{rho} draws r1's rows toward r{rho}'s prior; S{rho} draws r{rho}'s rows toward r1's."""
    for r in RATIOS_NEW:
        assert FIT_SOURCE[f"P{r}"] == ("r1", f"r{r}")
        assert FIT_SOURCE[f"S{r}"] == (f"r{r}", "r1")


def test_adjusted_preds_matches_argmax_logits_minus_log_prior() -> None:
    """argmax(log p - log pi) over stored probabilities agrees with argmax(logits - log pi)."""
    rng = np.random.default_rng(0)
    names = [f"c{i}" for i in range(N_CLASSES)]
    logits = rng.normal(size=(50, N_CLASSES))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    prior_counts = dict(zip(names, [50, 30, 20, 12, 7, 3, 1]))
    shares = np.log(
        np.asarray([prior_counts[n] for n in names], dtype=np.float64) / 123.0
    )
    expected = np.argmax(logits - shares, axis=1)
    actual = _adjusted_preds(probs, prior_counts, names)
    np.testing.assert_array_equal(actual, expected)
