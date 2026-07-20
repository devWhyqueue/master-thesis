"""Regressions for divergences found while reviewing the benchmark report.

Each test fails on the pre-fix implementation and pins the behavior required by
the benchmark protocol (`report/2_cls_imb_benchmark.tex`) and the companion
methods report (`../../1_cls_imb_methods/report/1_cls_imb_methods.tex`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from imbalance_benchmark.construction import effective_rho, _allocation_is_feasible
from imbalance_benchmark.manifest.pilot_training import stability_floor_from_curve
from imbalance_benchmark.modeling.losses import (
    SoftF1LossMulti,
    cfal_loss,
    _mix_ranked_bags,
)
from imbalance_benchmark.modeling.models import (
    AttentionMil,
    CfalPrototypeClassifier,
    OkoClassifier,
)
from imbalance_benchmark.modeling.oko import oko_set_loss, sample_oko_sets
from imbalance_benchmark.modeling.training import FIXED_BALANCED_SAMPLER_METHODS
from imbalance_benchmark.analysis.inference.context import _tail_classes
from imbalance_benchmark.analysis.inference.gates import (
    _SeverityInputs,
    _recovery_comparison,
    discrimination_gate_comparison,
)


# --- Finding 1: largest-feasible severity with disconnected feasibility ----------
def test_effective_rho_returns_largest_feasible_when_feasibility_is_disconnected():
    inventory = [372, 231, 107, 463, 114, 364, 96]
    floor, total = 20, 678
    # rho=10 is demonstrably feasible, so a severe (rho=100) request must not
    # collapse below the moderate rho=10 it already attains.
    assert _allocation_is_feasible(inventory, total, 10.0, floor)
    severe = effective_rho(inventory, 100.0, floor, total)
    moderate = effective_rho(inventory, 10.0, floor, total)
    assert severe >= 10.0
    assert severe >= moderate


# --- Finding 2: support-pilot BA rule must hold in all three orderings -----------
def test_stability_floor_requires_ba_increment_below_threshold_in_every_ordering():
    levels = [5, 10]
    # Mean BA increment is zero (+0.015, -0.015, 0), but two orderings each
    # exceed the 0.01 criterion, so level 5 is not stable.
    ba = {0: [0.5, 0.515], 1: [0.5, 0.485], 2: [0.5, 0.5]}
    recalls = {
        0: [[0.5, 0.5], [0.505, 0.505]],
        1: [[0.5, 0.5], [0.5, 0.5]],
        2: [[0.5, 0.5], [0.5, 0.5]],
    }
    assert stability_floor_from_curve(levels, ba, recalls) == 10


# --- Finding 3: CFAL diversity regularizer is finite for a binary task -----------
def test_cfal_loss_is_finite_for_two_classes():
    torch.manual_seed(0)
    model = CfalPrototypeClassifier(16, 8, 2, 0.0, 1.0)
    x = torch.randn(6, 16)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    loss = float(cfal_loss(model, x, y, np.array([3, 3])).detach())
    assert math.isfinite(loss)


# --- Finding 4: CFAL inference score is log-affinity, not raw affinity -----------
def test_cfal_forward_returns_log_affinities():
    torch.manual_seed(1)
    model = CfalPrototypeClassifier(16, 8, 3, 0.0, 1.0).eval()
    x = torch.randn(7, 16)
    forward = model(x)
    affinities = model.affinities(x)
    assert torch.all(forward <= 0.0)  # log of values in (0, 1]
    assert torch.all(affinities > 0.0) and torch.all(affinities <= 1.0)
    assert torch.allclose(forward, torch.log(affinities), atol=1e-6)


# --- Finding 4: CFAL objective uses margin 0.1 and un-renormalized 1/E_c weights -
def _cfal_reference(
    model: CfalPrototypeClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    counts: np.ndarray,
    gamma: float = 2.0,
    beta: float = 0.999,
    margin: float = 0.1,
) -> torch.Tensor:
    """Independent transcription of Eq. (CFAL) from the methods report."""
    eff = (1.0 - beta ** np.maximum(counts, 1.0)) / (1.0 - beta)
    inv_eff = torch.tensor(1.0 / eff, dtype=torch.float32)
    aff = model.affinities(x)
    true_aff = aff[torch.arange(len(y)), y]
    margins = torch.relu(margin + aff - true_aff.unsqueeze(1))
    margins = margins.masked_fill(F.one_hot(y, aff.shape[1]).bool(), 0.0).sum(dim=1)
    cls = (inv_eff[y] * (1.0 - true_aff).clamp(min=0.0).pow(gamma) * margins).mean()
    proto = F.normalize(model.prototypes, dim=-1, eps=1e-8)
    pw = (proto.unsqueeze(0) - proto.unsqueeze(1)).square().sum(dim=-1)
    reg = pw[torch.triu(torch.ones_like(pw), diagonal=1).bool()].var(unbiased=False)
    return cls + reg


def test_cfal_loss_matches_report_margin_and_weights():
    torch.manual_seed(2)
    model = CfalPrototypeClassifier(16, 8, 3, 0.0, 1.0).eval()
    x = torch.randn(9, 16)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    counts = np.array([50, 5, 1])
    assert torch.allclose(
        cfal_loss(model, x, y, counts), _cfal_reference(model, x, y, counts), atol=1e-6
    )


# --- Finding 5: Scholz hybrids use balanced oversampling; RankMix does not -------
def test_scholz_methods_are_the_balanced_sampler_hybrids():
    assert "ce_soft_f1" in FIXED_BALANCED_SAMPLER_METHODS
    assert "ce_soft_mcc" in FIXED_BALANCED_SAMPLER_METHODS
    assert "rankmix" not in FIXED_BALANCED_SAMPLER_METHODS


# --- Finding 5: soft-F1 uses one normalized multiclass probability vector --------
def test_soft_f1_is_softmax_normalized_not_independent_sigmoids():
    loss = SoftF1LossMulti(3)
    logits = torch.randn(5, 3)
    one_hot = F.one_hot(torch.tensor([0, 1, 2, 0, 1]), 3).float()
    # A constant added to every logit leaves the softmax distribution unchanged;
    # independent sigmoids would move, so the loss must be shift-invariant.
    assert torch.allclose(loss(logits, one_hot), loss(logits + 5.0, one_hot), atol=1e-6)


# --- Finding 6: RankMix keeps k = min(|B_a|, |B_b|) instances, not half ----------
def test_rankmix_keeps_min_bag_length_subsequence():
    torch.manual_seed(3)
    teacher = AttentionMil(16, 8, 2, 0.0)
    first, second = torch.randn(4, 16), torch.randn(6, 16)
    mixed = _mix_ranked_bags(teacher, first, second, 0, 1, 0.5)
    assert mixed.shape[0] == 4  # min(4, 6), not 4 // 2


# --- Finding 7: OKO draws k distinct odd classes without replacement -------------
def test_oko_odd_classes_are_distinct_and_exclude_the_pair_class():
    n_classes, k = 5, 3
    class_index = {c: [2 * c, 2 * c + 1] for c in range(n_classes)}  # idx // 2 == class
    pair, sets, _ = sample_oko_sets(class_index, n_classes, 64, k, np.random.default_rng(0))
    odd_classes = sets[:, 2:] // 2
    for row, pair_class in zip(odd_classes.tolist(), pair.tolist()):
        assert len(set(row)) == k
        assert pair_class not in row


# --- Finding 7: OKO aggregates the sum of per-example logits ---------------------
def test_oko_set_loss_sums_per_example_logits():
    torch.manual_seed(4)
    model = OkoClassifier(16, 8, 3, 0.0).eval()
    features = torch.randn(2 * 3, 16)
    pair = torch.tensor([0, 1])
    odd = torch.tensor([2, 0])
    encoded = model.encode(features).view(2, 3, -1)
    summed_logits_ref = F.cross_entropy(
        model.main_head(encoded).sum(dim=1), pair
    ) + F.cross_entropy(model.odd_head(encoded).sum(dim=1), odd)
    sum_embeddings = model.encode(features).view(2, 3, -1).sum(dim=1)
    old_formulation = F.cross_entropy(
        model.main_head(sum_embeddings), pair
    ) + F.cross_entropy(model.odd_head(sum_embeddings), odd)
    loss = oko_set_loss(model, features, batch_n=2, set_size=3, pair_labels=pair, odd_labels=odd)
    assert torch.allclose(loss, summed_logits_ref, atol=1e-6)
    # The biased head makes the two formulations genuinely differ.
    assert not torch.allclose(summed_logits_ref, old_formulation, atol=1e-4)


# --- Finding 8: tail tier is read from the analysed severity's allocation --------
def test_tail_classes_follow_the_analysed_severity_allocation():
    names = ["A", "B", "C", "D"]
    freeze = {
        "assignment_conditions": {
            "native": {
                "moderate": {"allocated_counts": {"A": 100, "B": 80, "C": 20, "D": 10}},
                "severe": {"allocated_counts": {"A": 100, "B": 10, "C": 80, "D": 90}},
            }
        },
        "tail_assignments": {"native": names},
    }
    moderate = _tail_classes(freeze, names, "native", "moderate")
    severe = _tail_classes(freeze, names, "native", "severe")
    assert moderate == [2, 3]  # C, D are the scarcest under moderate
    assert severe == [1, 2]  # B, C are the scarcest under severe
    assert moderate != severe


# --- Finding 9: gate and reported effect use the observed deficit, not its mean --
def test_gate_uses_observed_deficit_not_bootstrap_mean():
    # Replicate 0 is the observed cohort: the observed deficit is 0.01 (must not
    # open the 0.02 gate) while bootstrap replicates 1.. average ~0.03 with a CI
    # that excludes zero.
    balanced = np.array([0.51, 0.53, 0.54, 0.55])
    severity = np.array([0.50, 0.50, 0.51, 0.52])
    comparison, passed, dist = discrimination_gate_comparison(
        "moderate", balanced, severity
    )
    assert dist[0] == pytest.approx(0.01)  # observed deficit
    assert comparison["effect"] == pytest.approx(0.01)
    assert passed is False


# --- Finding 9: recovery is the observed numerator/denominator ratio -------------
def test_recovery_is_ratio_of_observed_points_not_mean_of_ratios():
    inp = _SeverityInputs({}, "moderate", {}, {}, None, 2, 10, 0, "native")
    # Index 0 is the observed cohort; ratio of observed points is 0.06/0.12 = 0.5,
    # whereas the mean of the two per-replicate ratios would be (0.2 + 5.0)/2 = 2.6.
    effect_dist = np.array([0.06, 0.02, 0.10])
    deficit_dist = np.array([0.12, 0.10, 0.02])
    entry = _recovery_comparison(
        inp,
        "weighted_ce",
        "discrimination",
        effect_dist,
        deficit_dist,
        gate_passed=True,
        p_value=None,
    )
    assert entry["recovery"] == pytest.approx(0.5)
    assert entry["numerator"] == pytest.approx(0.06)
    assert entry["denominator"] == pytest.approx(0.12)
