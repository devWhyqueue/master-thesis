from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from scripts.mil.bag_losses import (
    _mix_ranked_bags,
    _supervised_contrastive_loss,
)
from scripts.mil.bags import AttentionMil
from scripts.patch.losses import cfal_loss
from scripts.progan.core import ProgressiveGenerator
from scripts.progan.train import paper_batch_size
from scripts.patch.synthetic import _balance_target, _settings
from scripts.prep.patch_manifest import build_patch_manifest


def test_patch_manifest_uses_fixed_resolution_and_split(tmp_path: Path) -> None:
    slide_root = tmp_path / "A" / "0" / "slide-1"
    slide_root.mkdir(parents=True)
    for index in range(3):
        (slide_root / f"{index}.jpg").write_bytes(b"x")
    slides = pd.DataFrame([{"slide_id": "slide-1", "cancer_type": "A", "split": "train"}])
    frame = build_patch_manifest(tmp_path, slides, "0", 2)
    assert frame["split"].tolist() == ["train", "train"]
    assert frame["resolution"].tolist() == ["0", "0"]
    assert len(frame) == 2


def test_cfal_loss_is_zero_for_perfectly_separated_matching_prototypes() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    prototypes = embeddings.clone()
    weights = torch.ones(2)
    loss = cfal_loss(embeddings, targets, prototypes, weights, 0.01, 0.1, 2.0)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_sc_mil_reports_positive_pairs() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    targets = torch.tensor([0, 0, 1, 2])
    _, pairs = _supervised_contrastive_loss(embeddings, targets, 1.0)
    assert pairs == 2


def test_rankmix_preserves_ranked_feature_order_before_mixing() -> None:
    model = AttentionMil(2, 2, 2, 0.0)
    with torch.no_grad():
        model.instance_encoder[0].weight.copy_(torch.eye(2))
        model.instance_encoder[0].bias.zero_()
        model.classifier.weight.copy_(torch.eye(2))
        model.classifier.bias.zero_()
    first = torch.tensor([[0.0, 1.0], [4.0, 0.0], [3.0, 0.0]])
    second = torch.tensor([[0.0, 5.0], [0.0, 4.0]])
    mixed = _mix_ranked_bags(model, first, second, 0, 1, 0.5)
    expected = torch.tensor([[2.0, 2.5], [1.5, 2.0]])
    assert torch.allclose(mixed, expected)


def test_progan_matches_paper_batch_schedule_and_progressive_shape() -> None:
    assert [paper_batch_size(depth) for depth in range(1, 8)] == [
        64,
        64,
        32,
        16,
        4,
        2,
        1,
    ]
    generator = ProgressiveGenerator(latent_dim=8, max_depth=3, base_channels=32)
    noise = torch.randn(2, 8, 1, 1)
    assert generator(noise, depth=1, alpha=1.0).shape == (2, 3, 4, 4)
    assert generator(noise, depth=3, alpha=1.0).shape == (2, 3, 16, 16)


def test_progan_balances_to_head_patch_count() -> None:
    frame = pd.DataFrame({"cancer_type": ["A", "A", "A", "B"]})
    settings = _settings(
        {
            "patch_synthetic_progan": {
                "image_size": 256,
                "latent_dim": 8,
                "epochs_per_depth": 1,
                "learning_rate": 0.001,
                "beta1": 0.5,
                "max_real_patches_per_class": 8,
                "balance_target": "max_train_class_count",
                "fade_in_fraction": 0.5,
                "base_channels": 32,
                "max_classes": None,
            }
        }
    )
    assert _balance_target(frame, settings) == 3
