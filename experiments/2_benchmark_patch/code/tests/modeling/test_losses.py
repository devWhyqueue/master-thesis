from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from imbalance_benchmark.modeling.losses import (
    SoftF1LossMulti,
    cfal_loss,
)
from imbalance_benchmark.modeling.models import (
    AttentionMil,
    CfalPrototypeClassifier,
    OkoClassifier,
)
from imbalance_benchmark.modeling.oko import (
    oko_set_loss,
)
from imbalance_benchmark.modeling.training import _fit_step
import torch.nn.functional as F

DIM = 16

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

def test_supervised_contrastive_loss_reports_pairs_and_anchors():
    from imbalance_benchmark.modeling.losses import supervised_contrastive_loss

    embeddings = torch.randn(4, 8)
    targets = torch.tensor(
        [0, 0, 1, 1]
    )  # four directed positive pairs, four valid anchors

    loss, n_pairs, n_anchors = supervised_contrastive_loss(embeddings, targets, 0.1)

    assert n_pairs == 4  # (0,1),(1,0),(2,3),(3,2)
    assert n_anchors == 4
    assert torch.isfinite(loss)

    _, no_pairs, no_anchors = supervised_contrastive_loss(
        torch.randn(2, 8), torch.tensor([0, 1]), 0.1
    )
    assert no_pairs == 0 and no_anchors == 0

def test_oko_set_loss_finite_and_differentiable():
    model = OkoClassifier(DIM, 8, 3, 0.0)
    features = torch.randn(5 * 3, DIM)
    pair_labels = torch.tensor([0, 1, 2, 0, 1])
    odd_labels = torch.tensor([1, 2, 0, 2, 0])
    loss = oko_set_loss(
        model,
        features,
        batch_n=5,
        set_size=3,
        pair_labels=pair_labels,
        odd_labels=odd_labels,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.main_head.weight.grad is not None
    assert model.odd_head.weight.grad is not None

def test_cfal_loss_is_finite_for_two_classes():
    torch.manual_seed(0)
    model = CfalPrototypeClassifier(16, 8, 2, 0.0, 1.0)
    x = torch.randn(6, 16)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    loss = float(cfal_loss(model, x, y, np.array([3, 3])).detach())
    assert math.isfinite(loss)

def test_cfal_forward_returns_log_affinities():
    torch.manual_seed(1)
    model = CfalPrototypeClassifier(16, 8, 3, 0.0, 1.0).eval()
    x = torch.randn(7, 16)
    forward = model(x)
    affinities = model.affinities(x)
    assert torch.all(forward <= 0.0)  # log of values in (0, 1]
    assert torch.all(affinities > 0.0) and torch.all(affinities <= 1.0)
    assert torch.allclose(forward, torch.log(affinities), atol=1e-6)

def test_cfal_loss_matches_report_margin_and_weights():
    torch.manual_seed(2)
    model = CfalPrototypeClassifier(16, 8, 3, 0.0, 1.0).eval()
    x = torch.randn(9, 16)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    counts = np.array([50, 5, 1])
    assert torch.allclose(
        cfal_loss(model, x, y, counts), _cfal_reference(model, x, y, counts), atol=1e-6
    )

def test_soft_f1_is_softmax_normalized_not_independent_sigmoids():
    loss = SoftF1LossMulti(3)
    logits = torch.randn(5, 3)
    one_hot = F.one_hot(torch.tensor([0, 1, 2, 0, 1]), 3).float()
    # A constant added to every logit leaves the softmax distribution unchanged;
    # independent sigmoids would move, so the loss must be shift-invariant.
    assert torch.allclose(loss(logits, one_hot), loss(logits + 5.0, one_hot), atol=1e-6)

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

def test_cfal_tuned_sigma_is_not_reused_as_a_loss_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CFAL's only method-specific grid control is its affinity bandwidth."""
    seen: dict[str, object] = {}

    def fixed_loss(
        model: torch.nn.Module,
        features: torch.Tensor,
        targets: torch.Tensor,
        class_counts: np.ndarray,
    ) -> torch.Tensor:
        seen["called"] = True
        return model(features).sum()

    monkeypatch.setattr("imbalance_benchmark.modeling.training.cfal_loss", fixed_loss)
    model = torch.nn.Linear(2, 2)
    loss = _fit_step(
        {"features": torch.ones(2, 2), "target": torch.tensor([0, 1])},
        {
            "is_mil": False,
            "method": "cfal",
            "device": torch.device("cpu"),
            "model": model,
            "criterion": torch.nn.CrossEntropyLoss(),
            "param": 4.0,
            "class_counts": np.array([1, 1]),
        },
        0,
        1,
    )

    assert seen["called"] is True
    assert loss.requires_grad
