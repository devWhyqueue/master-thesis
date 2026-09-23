"""Unit tests for the gate-4 power projection: monotone in effect size and residual spread."""

from __future__ import annotations

import numpy as np

from separation.precision import projected_power


def test_projected_power_is_high_with_no_residual_and_a_real_effect():
    """Zero pilot residual and the pre-registered 2 pp effect should project near-certain power."""
    power = projected_power(np.zeros(3), n_sim=500, n_bootstrap=200)
    assert power > 0.99


def test_projected_power_is_low_when_residual_swamps_the_effect():
    """A residual spread far larger than the effect should project low power."""
    power = projected_power(
        np.array([-50.0, 0.0, 50.0]), n_sim=500, n_bootstrap=200
    )
    assert power < 0.5


def test_projected_power_increases_with_effect_size():
    """A larger assumed effect, same residual, never projects less power."""
    residual = np.array([-1.0, 0.5, 0.5])
    small = projected_power(residual, n_sim=1000, n_bootstrap=200, effect_pp=0.5)
    large = projected_power(residual, n_sim=1000, n_bootstrap=200, effect_pp=5.0)
    assert large >= small
