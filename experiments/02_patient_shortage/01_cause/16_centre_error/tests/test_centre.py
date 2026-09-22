"""Unit tests for centre manipulations, whitening, noise, sharding, and derived shares."""

from __future__ import annotations

import numpy as np
import pytest

from centre import ARMS, N_DRAWS, N_SPLITS
from centre.analyze import derived
from centre.arms import (
    between_patient_eigenbasis,
    centre_error_parts,
    class_means,
    discriminant_basis,
    move_centres,
    noise_shift,
    whiten,
)
from centre.fit import decode_shard_index, shard_count, split_arm


def _toy(seed: int = 0, n: int = 60, c: int = 4, d: int = 7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), np.arange(n) % c, rng.normal(size=(c, d))


def test_move_centres_hits_target_and_keeps_offsets():
    """Shifted class means equal the target; offsets from the class mean are unchanged."""
    x, y, target = _toy()
    moved = move_centres(x, y, target)
    np.testing.assert_allclose(class_means(moved, y, 4), target, atol=1e-12)
    np.testing.assert_allclose(
        moved - target[y], x - class_means(x, y, 4)[y], atol=1e-12
    )


def test_centre_error_parts_sum_to_error_and_split_shared():
    """Shared + discriminant + off-discriminant = error; class-specific parts average to zero."""
    _, _, centres = _toy(1)
    error = np.random.default_rng(2).normal(size=centres.shape)
    parts = centre_error_parts(error, discriminant_basis(centres))
    np.testing.assert_allclose(sum(parts), error, atol=1e-12)
    np.testing.assert_allclose(parts.discriminant.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(parts.off_discriminant @ discriminant_basis(centres).T, 0.0, atol=1e-12)


def test_whiten_matches_explicit_inverse_square_root():
    """Low-rank whitening equals x (I + kappa B)^(-1/2) formed explicitly."""
    rng = np.random.default_rng(3)
    dev, x = rng.normal(size=(5, 8)), rng.normal(size=(10, 8))
    basis, eigvals = between_patient_eigenbasis(dev, dof=4)
    b = dev.T @ dev / 4
    vals, vecs = np.linalg.eigh(np.eye(8) + 2.5 * b)
    explicit = x @ (vecs * vals**-0.5) @ vecs.T
    np.testing.assert_allclose(whiten(x, basis, eigvals, 2.5), explicit, atol=1e-10)


def test_noise_shift_covariance_is_sigma_over_g():
    """Sampled centre errors have covariance deviations.T deviations / ((n - 1) g)."""
    rng = np.random.default_rng(4)
    dev = rng.normal(size=(12, 3))
    dev -= dev.mean(axis=0)
    draws = np.stack([noise_shift(dev, 5, rng) for _ in range(40000)])
    expected = dev.T @ dev / (11 * 5)
    np.testing.assert_allclose(np.cov(draws.T), expected, atol=0.02 * np.abs(expected).max())


def test_shards_cover_every_split_and_draw_once():
    """Shard indices map one-to-one onto (split, draw) pairs."""
    decoded = {decode_shard_index(i) for i in range(shard_count())}
    assert decoded == {(s, d) for s in range(N_SPLITS) for d in range(N_DRAWS)}
    with pytest.raises(ValueError):
        decode_shard_index(shard_count())


def test_arm_names_parse_and_shares_follow_gaps():
    """Arm names decode to (family, G); a halved centre gap gives share 0.5."""
    assert split_arm("CW10") == ("CW", 10) and split_arm("Swap5") == ("Swap", 5)
    base = {"R": (60.0, 66.0, 70.0), "C": (70.0, 73.0, 75.0), "N": (61.0, 66.0, 70.0), "CW": (72.0, 74.0, 75.0)}
    arm = {f"{f}{g}": np.array([v]) for f, vals in base.items() for g, v in zip((5, 10, 20), vals)}
    arm |= {name: np.array([60.0]) for name in ARMS if name not in arm}
    out = derived(arm)
    assert out["centre_share_5_to_10"][0] == pytest.approx(0.5)
    assert out["combined_share_5_to_20"][0] == pytest.approx(0.7)
    assert out["C5_minus_R20"][0] == pytest.approx(0.0)
