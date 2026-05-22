from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from scripts.mil.bag_losses import (
    _mix_ranked_bags,
    _supervised_contrastive_loss,
)
from scripts.mil.bags import AttentionMil, _feature_to_bag
from scripts.patch.artifacts import (
    load_training_checkpoint,
    save_patch_checkpoint,
    save_training_checkpoint,
    seed_patch_run,
)
from scripts.patch.data import PatchImageDataset, patch_loader, uses_balanced_sampler
from scripts.staging.io import stage_destination
from scripts.patch.models import PatchClassifier
from scripts.patch.losses import ScholzCombinedLoss, SoftF1LossMulti, SoftMCCLossMulti
from scripts.progan.core import (
    ProgressiveDiscriminator,
    ProgressiveGenerator,
    ProGanSettings,
)
from scripts.progan.train import paper_batch_size
from scripts.common import ensure_dirs
from scripts.progan.manifest import (
    _class_image_paths,
    balance_target as _balance_target,
    decode_progan_array_task,
    progan_array_upper_bound,
    progan_settings as _settings,
)
from scripts.progan.storage import generated_counts_match as _generated_counts_match
from scripts.training.support_tiers import class_tier_labels
from scripts.prep.patch_manifest import build_patch_manifest
from scripts.tuning.aggregate import _select_all
from scripts.tuning.grid import task_count, task_for_array_index, validate_tuning_params
from scripts.tuning.paths import tuning_result_dir


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


def test_stage_destination_preserves_raw_layout(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "A" / "0" / "slide" / "1.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x")
    stage_dir = tmp_path / "stage"
    destination = stage_destination(source, stage_dir, raw_root)
    assert destination == stage_dir / "raw" / "A" / "0" / "slide" / "1.jpg"


def test_training_checkpoint_resumes_from_next_epoch(tmp_path: Path) -> None:
    model = PatchClassifier(4, 2, 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    save_training_checkpoint(
        tmp_path, "patch_ce", 1, model, optimizer, ["A", "B"], epoch=7, epochs=30
    )
    resumed_model = PatchClassifier(4, 2, 0.0)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    next_epoch, class_names = load_training_checkpoint(
        tmp_path / "checkpoint_latest.pt",
        resumed_model,
        resumed_optimizer,
        torch.device("cpu"),
    )
    assert next_epoch == 8
    assert class_names == ["A", "B"]
    for first, second in zip(model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.allclose(first, second)


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


def test_wsi_bag_cap_uses_evenly_spaced_instances(tmp_path: Path) -> None:
    path = tmp_path / "features.pt"
    features = torch.arange(50, dtype=torch.float32).reshape(50, 1)
    torch.save(features, path)
    bag = _feature_to_bag(str(path), max_instances=30)
    expected_indices = torch.linspace(0, 49, 30).long()
    assert bag.shape == (30, 1)
    assert torch.equal(bag.squeeze(1), features[expected_indices].squeeze(1))


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


def test_progan_subsamples_real_patches_with_stable_seed() -> None:
    frame = pd.DataFrame(
        {
            "cancer_type": ["A"] * 10,
            "image_path": [f"/data/raw/A/0/slide/{index}.jpg" for index in range(10)],
        }
    )
    settings = ProGanSettings(
        image_size=256,
        latent_dim=8,
        epochs_per_depth=1,
        learning_rate=0.001,
        beta1=0.5,
        max_real_patches_per_class=4,
        balance_target="max_train_class_count",
        max_classes=None,
        fade_in_fraction=0.5,
        base_channels=32,
    )
    raw_root = Path("/data/raw")
    first = _class_image_paths(frame, "A", settings, raw_root, benchmark_seed=1)
    second = _class_image_paths(frame, "A", settings, raw_root, benchmark_seed=1)
    other_seed = _class_image_paths(frame, "A", settings, raw_root, benchmark_seed=2)
    assert len(first) == 4
    assert first == second
    assert first != other_seed


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


def test_support_tiers_use_dataset_slide_counts() -> None:
    class_names = [f"class_{index}" for index in range(32)]
    slide_counts = {name: index + 1 for index, name in enumerate(class_names)}
    labels = class_tier_labels(class_names, slide_counts)
    assert labels["class_0"] == "tail"
    assert labels["class_7"] == "tail"
    assert labels["class_8"] == "body"
    assert labels["class_23"] == "body"
    assert labels["class_24"] == "head"
    assert labels["class_31"] == "head"


def test_tuning_grid_expands_expected_array_sizes() -> None:
    assert task_count("patch_feature") == 57
    assert task_count("wsi_bag") == 57
    first_variant, first_seed = task_for_array_index("patch_feature", 0)
    last_variant, last_seed = task_for_array_index("wsi_bag", 56)
    assert first_variant.method == "patch_feature_weighted_ce"
    assert first_seed == 0
    assert last_variant.method == "sc_mil"
    assert last_seed == 2


def test_tuning_validation_rejects_unsupported_parameters() -> None:
    validate_tuning_params("patch_feature", "patch_feature_focal", {"focal_gamma": 1.5})
    with pytest.raises(ValueError, match="Unsupported tuning parameter"):
        validate_tuning_params(
            "patch_feature", "patch_feature_focal", {"rankmix_alpha": 0.5}
        )


def test_tuning_result_paths_do_not_overwrite_fixed_outputs(tmp_path: Path) -> None:
    paths = {
        "root": tmp_path,
        "results": tmp_path / "outputs" / "results",
        "wsi_results": tmp_path / "outputs" / "results_wsi_bag",
    }
    fixed_patch = paths["results"] / "patch_feature" / "patch_feature_focal" / "seed=1"
    fixed_wsi = paths["wsi_results"] / "mil_focal" / "seed=1"
    tuned_patch = tuning_result_dir(
        paths, "patch_feature", "patch_feature_focal", "focal_gamma=1", 1
    )
    tuned_wsi = tuning_result_dir(paths, "wsi_bag", "mil_focal", "focal_gamma=1", 1)
    assert fixed_patch != tuned_patch
    assert fixed_wsi != tuned_wsi
    assert "outputs/tuning/patch_feature" in tuned_patch.as_posix()
    assert "outputs/tuning/wsi_bag" in tuned_wsi.as_posix()


def test_tuning_selection_requires_complete_seed_sets() -> None:
    frame = pd.DataFrame(
        [
            {
                "benchmark": "patch_feature",
                "method": "patch_feature_focal",
                "variant": "focal_gamma=1",
                "params": '{"focal_gamma":1}',
                "method_label": "Focal",
                "baseline_test_macro_f1": 0.4,
                "fixed_test_macro_f1": 0.3,
                "fixed_test_balanced_accuracy": 0.4,
                "val_macro_f1": 0.5,
                "val_balanced_accuracy": 0.6,
                "test_macro_f1": 0.45,
                "test_balanced_accuracy": 0.5,
                "test_accuracy": 0.7,
            }
        ]
    )
    with pytest.raises(ValueError, match="Incomplete tuning results"):
        _select_all(frame, allow_incomplete=False)
