from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from scripts.mil.bag_losses import (
    _mix_ranked_bags,
    _supervised_contrastive_loss,
)
from scripts.mil.bags import AttentionMil
from scripts.patch.artifacts import save_patch_checkpoint, seed_patch_run
from scripts.patch.data import PatchImageDataset, patch_loader, uses_balanced_sampler
from scripts.patch.models import PatchClassifier
from scripts.patch.losses import ScholzCombinedLoss, SoftF1LossMulti, SoftMCCLossMulti
from scripts.progan.core import ProgressiveDiscriminator, ProgressiveGenerator
from scripts.progan.train import paper_batch_size
from scripts.common import ensure_dirs
from scripts.progan.manifest import (
    balance_target as _balance_target,
    decode_progan_array_task,
    progan_array_upper_bound,
    progan_settings as _settings,
)
from scripts.progan.storage import generated_counts_match as _generated_counts_match
from scripts.prep.patch_manifest import build_patch_manifest


def test_patch_manifest_uses_fixed_resolution_and_split(tmp_path: Path) -> None:
    slide_root = tmp_path / "A" / "0" / "slide-1"
    slide_root.mkdir(parents=True)
    for index in range(3):
        (slide_root / f"{index}.jpg").write_bytes(b"x")
    slides = pd.DataFrame(
        [{"slide_id": "slide-1", "cancer_type": "A", "split": "train"}]
    )
    frame = build_patch_manifest(tmp_path, slides, "0", 2)
    assert frame["split"].tolist() == ["train", "train"]
    assert frame["resolution"].tolist() == ["0", "0"]
    assert len(frame) == 2


def test_scholz_combined_loss_decreases_when_predictions_improve() -> None:
    targets = torch.tensor([0, 1, 2])
    weak_logits = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
        ]
    )
    strong_logits = torch.tensor(
        [
            [5.0, -5.0, -5.0],
            [-5.0, 5.0, -5.0],
            [-5.0, -5.0, 5.0],
        ]
    )
    loss_fn = ScholzCombinedLoss(3, "f1")
    assert loss_fn(strong_logits, targets).item() < loss_fn(weak_logits, targets).item()


def test_soft_f1_loss_decreases_when_predictions_improve() -> None:
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    weak_logits = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
        ]
    )
    strong_logits = torch.tensor(
        [
            [5.0, -5.0, -5.0],
            [-5.0, 5.0, -5.0],
            [-5.0, -5.0, 5.0],
        ]
    )
    loss_fn = SoftF1LossMulti(3)
    assert loss_fn(strong_logits, labels).item() < loss_fn(weak_logits, labels).item()


def test_soft_mcc_loss_decreases_when_predictions_improve() -> None:
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    weak_logits = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
        ]
    )
    strong_logits = torch.tensor(
        [
            [5.0, -5.0, -5.0],
            [-5.0, 5.0, -5.0],
            [-5.0, -5.0, 5.0],
        ]
    )
    loss_fn = SoftMCCLossMulti()
    assert loss_fn(strong_logits, labels).item() < loss_fn(weak_logits, labels).item()


def test_patch_seed_reproduces_model_initialization() -> None:
    seed_patch_run(7)
    first_model = PatchClassifier(4, 3, 0.0)
    seed_patch_run(7)
    second_model = PatchClassifier(4, 3, 0.0)
    for first, second in zip(
        first_model.parameters(), second_model.parameters(), strict=True
    ):
        assert torch.allclose(first, second)


def test_scholz_methods_use_balanced_sampler() -> None:
    assert uses_balanced_sampler("patch_ce_soft_f1_balanced")
    assert uses_balanced_sampler("patch_ce_soft_mcc_balanced")
    assert not uses_balanced_sampler("patch_ce")


def test_scholz_patch_loader_uses_weighted_sampler(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "cancer_type": ["A", "A", "B"],
            "image_path": [
                str(tmp_path / "a0.jpg"),
                str(tmp_path / "a1.jpg"),
                str(tmp_path / "b0.jpg"),
            ],
        }
    )
    for path in frame["image_path"]:
        Path(path).write_bytes(b"x")
    dataset = PatchImageDataset(frame, {"A": 0, "B": 1}, 8)
    labels = np.array([0, 0, 1], dtype=np.int64)
    from torch.utils.data import WeightedRandomSampler

    loader = patch_loader(dataset, labels, "patch_ce_soft_f1_balanced", 2, 0, 0)
    assert isinstance(loader.sampler, WeightedRandomSampler)


def test_patch_checkpoint_stores_model_state(tmp_path: Path) -> None:
    model = PatchClassifier(4, 2, 0.0)
    save_patch_checkpoint(tmp_path, "patch_ce_soft_f1_balanced", 3, model, ["A", "B"])
    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu")
    assert checkpoint["class_names"] == ["A", "B"]
    assert checkpoint["seed"] == 3
    assert "model_state_dict" in checkpoint


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


def test_progan_discriminator_accepts_minibatch_stddev_channel() -> None:
    discriminator = ProgressiveDiscriminator(max_depth=3, base_channels=32)
    images = torch.randn(2, 3, 16, 16)
    assert discriminator(images, depth=3, alpha=1.0).shape == (2,)


def test_progan_cache_validation_rejects_under_counted_class() -> None:
    rows = [
        {"cancer_type": "A", "image_path": "A_0.jpg"},
        {"cancer_type": "B", "image_path": "B_0.jpg"},
    ]
    assert not _generated_counts_match(rows, {"A": 2, "B": 1})


def test_progan_array_upper_bound_scales_with_tail_classes(tmp_path: Path) -> None:
    config = {
        "paths": {
            "tcga_root": str(tmp_path),
            "raw_root": str(tmp_path),
            "feature_dir": str(tmp_path),
            "outputs": str(tmp_path / "outputs"),
        },
        "patch_training": {"seeds": [0]},
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
            "max_classes": 2,
        },
    }
    frame = pd.DataFrame(
        {
            "cancer_type": ["A", "A", "A", "B", "C"],
            "split": ["train"] * 5,
            "image_path": [f"{idx}.jpg" for idx in range(5)],
        }
    )
    paths = ensure_dirs(config)
    frame.to_csv(paths["data"] / "patch_manifest_seed=0.csv", index=False)
    assert progan_array_upper_bound(config) == 1


def test_decode_progan_array_task_maps_seed_and_class(tmp_path: Path) -> None:
    config = {
        "paths": {
            "tcga_root": str(tmp_path),
            "raw_root": str(tmp_path),
            "feature_dir": str(tmp_path),
            "outputs": str(tmp_path / "outputs"),
        },
        "patch_training": {"seeds": [0, 1, 2]},
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
            "max_classes": 2,
        },
    }
    frame = pd.DataFrame(
        {
            "cancer_type": ["A", "A", "A", "B", "C"],
            "split": ["train"] * 5,
            "image_path": [f"{idx}.jpg" for idx in range(5)],
        }
    )
    paths = ensure_dirs(config)
    frame.to_csv(paths["data"] / "patch_manifest_seed=0.csv", index=False)
    assert decode_progan_array_task(config, 0, smoke=True) == (0, "B")
    assert decode_progan_array_task(config, 9, smoke=True) is None


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
