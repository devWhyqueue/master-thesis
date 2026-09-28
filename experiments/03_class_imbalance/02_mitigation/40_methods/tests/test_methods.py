"""Unit tests for the mitigation methods' sampling, losses, and heads (CPU, synthetic)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mitigation.methods import (
    CosineHead,
    DisAlign,
    augment,
    cuda_batch,
    disalign_class_weights,
    gcl_delta,
    gcl_logits,
    la_loss,
    reinit_linear_head,
    sample_weights,
    update_levels,
)

_Y = np.array([0, 0, 0, 1], dtype=np.int64)


def test_power_sampling_uniform_at_s0() -> None:
    """s=0 recovers empirical (uniform per-row) sampling."""
    weights = sample_weights(_Y, 2, "power", 0.0)
    assert np.allclose(weights, weights[0])


def test_power_sampling_balanced_at_s1() -> None:
    """s=1 gives every class the same expected total mass."""
    weights = sample_weights(_Y, 2, "power", 1.0)
    assert weights[_Y == 0].sum() == pytest.approx(weights[_Y == 1].sum())


def test_cuda_batch_p_aug_zero_is_identity() -> None:
    """p_aug=0 leaves every row unchanged, whatever the per-row levels."""
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8)
    levels = torch.tensor([5, 3, 0, 8])
    out = cuda_batch(images, levels, 0.0, max_strength=10, generator=generator)
    assert torch.equal(out, images)


def test_cuda_batch_strength_zero_is_identity() -> None:
    """Rows at level 0 are unchanged even when always selected for augmentation."""
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8)
    levels = torch.zeros(4, dtype=torch.long)
    out = cuda_batch(images, levels, 1.0, max_strength=10, generator=generator)
    assert torch.equal(out, images)


def test_augment_strength_zero_is_identity() -> None:
    """strength=0 leaves the image unchanged (Eq. cuda-aug)."""
    generator = torch.Generator().manual_seed(0)
    img = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
    assert torch.equal(augment(img, 0, max_strength=10, generator=generator), img)


def test_update_levels_rises_when_always_correct() -> None:
    """A class whose checks all pass climbs by one level (Eq. cuda-lol)."""
    generator = torch.Generator().manual_seed(0)
    rows = torch.randint(0, 255, (20, 3, 32, 32), dtype=torch.uint8)
    levels = torch.tensor([3])

    def always_correct(batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros(batch.shape[0], dtype=torch.long)

    new_levels = update_levels(
        levels, [rows], always_correct, gamma=0.6, check_size=10, max_strength=10,
        generator=generator,
    )
    assert new_levels[0].item() == 4


def test_update_levels_falls_when_always_wrong() -> None:
    """A class whose checks all fail drops by one level (Eq. cuda-lol)."""
    generator = torch.Generator().manual_seed(0)
    rows = torch.randint(0, 255, (20, 3, 32, 32), dtype=torch.uint8)
    levels = torch.tensor([3])

    def always_wrong(batch: torch.Tensor) -> torch.Tensor:
        return torch.full((batch.shape[0],), 1, dtype=torch.long)

    new_levels = update_levels(
        levels, [rows], always_wrong, gamma=0.6, check_size=10, max_strength=10,
        generator=generator,
    )
    assert new_levels[0].item() == 2


def test_update_levels_clipped_at_bounds() -> None:
    """Level 0 never drops below 0; level S never rises above S."""
    generator = torch.Generator().manual_seed(0)
    rows = torch.randint(0, 255, (20, 3, 32, 32), dtype=torch.uint8)

    def always_wrong(batch: torch.Tensor) -> torch.Tensor:
        return torch.full((batch.shape[0],), 1, dtype=torch.long)

    def always_correct(batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros(batch.shape[0], dtype=torch.long)

    at_floor = update_levels(
        torch.tensor([0]), [rows], always_wrong, gamma=0.6, check_size=10,
        max_strength=10, generator=generator,
    )
    at_ceiling = update_levels(
        torch.tensor([10]), [rows], always_correct, gamma=0.6, check_size=10,
        max_strength=10, generator=generator,
    )
    assert at_floor[0].item() == 0
    assert at_ceiling[0].item() == 10


def test_la_loss_tau_zero_is_ce() -> None:
    """tau=0 recovers plain CE (no prior offset)."""
    logits = torch.randn(4, 3)
    y = torch.tensor([0, 1, 2, 0])
    log_prior = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    assert torch.isclose(la_loss(logits, y, log_prior, 0.0), F.cross_entropy(logits, y))


def test_gcl_logits_sigma_zero_is_scaled_cosine() -> None:
    """sigma=0 recovers s*cos(theta) with no noise."""
    cos_theta = torch.rand(4, 3)
    delta = torch.tensor([0.0, 1.0, 2.0])
    out = gcl_logits(cos_theta, delta, s=30.0, sigma=0.0)
    assert torch.equal(out, 30.0 * cos_theta)


def test_gcl_logits_head_class_gets_no_noise() -> None:
    """The class with delta_j=0 (most frequent) is never perturbed, any sigma."""
    torch.manual_seed(0)
    cos_theta = torch.rand(5, 3)
    delta = torch.tensor([0.0, 1.0, 2.0])
    out = gcl_logits(cos_theta, delta, s=30.0, sigma=2.0)
    assert torch.equal(out[:, 0], 30.0 * cos_theta[:, 0])
    assert not torch.equal(out[:, 1], 30.0 * cos_theta[:, 1])


def test_disalign_zero_init_is_identity() -> None:
    """Zero-initialized alpha/beta always recovers the frozen head's raw logits."""
    calibration = DisAlign(dim=6, num_classes=3)
    features = torch.randn(5, 6)
    logits_o = torch.randn(5, 3)
    assert torch.allclose(calibration(features, logits_o), logits_o)


def test_gcl_sampler_between_empirical_and_balanced() -> None:
    """The effective-number sampler's minority share sits between empirical and balanced."""
    y = np.array([0] * 90 + [1] * 10, dtype=np.int64)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    delta = gcl_delta(counts)
    empirical_minority = sample_weights(y, 2, "power", 0.0)[y == 1].sum()
    balanced_minority = sample_weights(y, 2, "power", 1.0)[y == 1].sum()
    gcl_minority = sample_weights(y, 2, "gcl", delta=delta, a=0.9, b=0.09)[y == 1].sum()
    assert empirical_minority < gcl_minority < balanced_minority


def test_crt_reinit_changes_head_only() -> None:
    """Reinit mutates the head's own weight/bias tensors and leaves other modules untouched."""
    head = nn.Linear(4, 2)
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, 1.0)
    other = nn.Linear(4, 2)
    other_weight_before = other.weight.detach().clone()
    reinit_linear_head(head)
    assert not torch.equal(head.weight, torch.zeros_like(head.weight))
    assert torch.equal(head.bias, torch.zeros_like(head.bias))
    assert torch.equal(other.weight, other_weight_before)


def test_disalign_class_weights_rho_zero_is_uniform() -> None:
    """rho=0 weights every patch equally (Eq. disalign-loss)."""
    weights = disalign_class_weights(np.array([90.0, 10.0]), 0.0)
    assert np.allclose(weights, [0.5, 0.5])


def test_cosine_head_returns_unit_range() -> None:
    """CosineHead output is a cosine similarity, bounded in [-1, 1]."""
    head = CosineHead(dim=8, num_classes=4)
    out = head(torch.randn(6, 8))
    assert torch.all(out <= 1.0 + 1e-5) and torch.all(out >= -1.0 - 1e-5)
