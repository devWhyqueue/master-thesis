"""Unit tests for exp-39's phase-04 decomposition and bootstrap-support helpers.

These target the pure numeric logic (``_decomposition``, interval packing, the
class-support check) directly on synthetic arm-accuracy distributions, without
touching run records, features, or a real bootstrap context.
"""

from __future__ import annotations

import numpy as np

from transfer import ARMS, RATIOS_NEW
from transfer.analyze.accuracy import bonferroni_interval, decomposition, pack
from transfer.analyze.bootstrap import class_support_rejections

_N_REPLICATES = 200


def _flat_ba(value: float) -> np.ndarray:
    """A constant (n_replicates,) distribution: every replicate equals ``value``."""
    return np.full(_N_REPLICATES, value, dtype=np.float64)


def _ba(values: dict[str, float]) -> dict[str, np.ndarray]:
    return {arm: _flat_ba(values.get(arm, 50.0)) for arm in ARMS}


def test_decomposition_identity_holds_for_every_encoder_rho() -> None:
    rng = np.random.default_rng(0)
    ba = {
        m: {arm: rng.normal(60, 5, size=_N_REPLICATES) for arm in ARMS}
        for m in ("virchow2", "uni2h")
    }
    dists = decomposition(ba)
    for m in ("virchow2", "uni2h"):
        for r in RATIOS_NEW:
            d_r = dists[f"DR_{m}_{r}"]
            d_p = dists[f"DP_{m}_{r}"]
            d_s = dists[f"DS_{m}_{r}"]
            i = dists[f"I_{m}_{r}"]
            np.testing.assert_allclose(i, d_r - d_p - d_s)


def test_identical_encoders_give_exactly_zero_damage_contrast() -> None:
    """Swapping features for a copy of the same encoder: zero encoder damage contrast."""
    rng = np.random.default_rng(1)
    shared = {arm: rng.normal(55, 4, size=_N_REPLICATES) for arm in ARMS}
    ba = {"virchow2": shared, "uni2h": shared}
    dists = decomposition(ba)
    for r in RATIOS_NEW:
        for x in ("DR", "DP", "DS", "I"):
            np.testing.assert_array_equal(dists[f"delta_{x}_{r}"], 0.0)
    np.testing.assert_array_equal(dists["delta_B"], 0.0)


def test_known_toy_values_give_expected_sign_and_pp_units() -> None:
    """Hand-computed damages: UNI2-h damaged less than Virchow2 at rho=100."""
    v2 = _ba({"B": 80.0, "R100": 60.0, "P100": 75.0, "S100": 65.0})
    uni = _ba({"B": 80.0, "R100": 70.0, "P100": 78.0, "S100": 72.0})
    dists = decomposition({"virchow2": v2, "uni2h": uni})
    # D_R = B - R100: virchow2 20pp, uni2h 10pp.
    np.testing.assert_allclose(dists["DR_virchow2_100"], 20.0)
    np.testing.assert_allclose(dists["DR_uni2h_100"], 10.0)
    # delta_DR = D_R(uni2h) - D_R(virchow2): negative means UNI2-h reduces damage.
    np.testing.assert_allclose(dists["delta_DR_100"], -10.0)
    assert float(dists["delta_DR_100"][0]) < 0


def test_bonferroni_interval_is_wider_than_a_95pct_interval() -> None:
    rng = np.random.default_rng(2)
    dist = np.concatenate([[0.0], rng.normal(0, 1, size=5000)])
    lo975, hi975 = bonferroni_interval(dist)
    lo95, hi95 = np.nanpercentile(dist[1:], 2.5), np.nanpercentile(dist[1:], 97.5)
    assert lo975 < lo95 < hi95 < hi975


def test_pack_adds_primary_bounds_only_when_requested() -> None:
    dist = np.concatenate([[1.0], np.random.default_rng(3).normal(1, 1, size=1000)])
    secondary = pack(dist, primary=False)
    primary = pack(dist, primary=True)
    assert "ci_1_25" not in secondary
    assert {"ci_1_25", "ci_98_75"} <= primary.keys()


def test_class_support_rejections_counts_absent_classes() -> None:
    labels = np.array([0, 0, 1, 1, 1])
    assert class_support_rejections(labels, n_classes=2) == 0
    assert class_support_rejections(labels, n_classes=3) == 1
