from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.datasets.data import BagFeatureDataset, ImbalanceDataset, bag_collate
from imbalance_benchmark.modeling.context import GRIDS, LEARNING_RATE_GRID, get_grid_configs, roster_for_regime
from imbalance_benchmark.modeling.losses import rankmix_bag_loss
from imbalance_benchmark.modeling.models import (
    AttentionMil,
    CfalPrototypeClassifier,
    DualExpertMil,
    MLP,
    OkoClassifier,
    build_model,
)
from imbalance_benchmark.modeling.oko import build_class_index, oko_set_loss, sample_oko_sets
from imbalance_benchmark.modeling.special_methods import fit_crt, fit_method, mde_bag_loss
from imbalance_benchmark.modeling.training import ClassAwareBatchSampler, update_budget

DIM = 16


def _write_patch_manifest(tmp_path: Path, n_classes: int = 3, per_class: int = 8) -> Path:
    classes = [f"class_{i}" for i in range(n_classes)]
    rows = []
    for cls in classes:
        for p in range(per_class):
            slide_id = f"{cls}_SLIDE_{p}"
            feature_path = tmp_path / f"{slide_id}.pt"
            torch.save(torch.randn(1, DIM), feature_path)
            rows.append(
                {
                    "case_id": f"{cls}_PAT_{p}",
                    "slide_id": slide_id,
                    "cancer_type": cls,
                    "feature_path": str(feature_path),
                }
            )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _write_bag_manifest(tmp_path: Path, n_classes: int = 3, per_class: int = 6) -> Path:
    classes = [f"class_{i}" for i in range(n_classes)]
    rows = []
    for cls in classes:
        for p in range(per_class):
            slide_id = f"{cls}_SLIDE_{p}"
            feature_path = tmp_path / f"{slide_id}.pt"
            torch.save(torch.randn(5, DIM), feature_path)
            rows.append(
                {
                    "case_id": f"{cls}_PAT_{p}",
                    "slide_id": slide_id,
                    "cancer_type": cls,
                    "feature_path": str(feature_path),
                }
            )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _patch_ctx(method: str, tmp_path: Path, param: float | None = None, n_classes: int = 3) -> dict:
    manifest = _write_patch_manifest(tmp_path, n_classes=n_classes)
    train_ds = ImbalanceDataset(manifest)
    model_kwargs = {"input_dim": DIM, "hidden_dim": 8, "dropout": 0.0}

    def model_factory() -> torch.nn.Module:
        return build_model(method, False, n_classes=n_classes, param=param, **model_kwargs)

    return {
        "method": method,
        "model": model_factory(),
        "model_factory": model_factory,
        "train_dataset": train_ds,
        "val_loader": torch.utils.data.DataLoader(train_ds, batch_size=8),
        "device": torch.device("cpu"),
        "config": {"patch_training": {"batch_size": 8}},
        "param_config": {"lr": 1e-3, "parameter": param} if param is not None else {"lr": 1e-3},
        "seed": 0,
        "is_mil": False,
        "n_classes": n_classes,
        "train_labels": train_ds.get_int_targets(),
    }


def _bag_ctx(method: str, tmp_path: Path, param: float | None = None, n_classes: int = 3) -> dict:
    manifest = _write_bag_manifest(tmp_path, n_classes=n_classes)
    train_ds = BagFeatureDataset(manifest, max_instances=5)
    model_kwargs = {"input_dim": DIM, "hidden_dim": 8, "dropout": 0.0}

    def model_factory() -> torch.nn.Module:
        return build_model(method, True, n_classes=n_classes, param=param, **model_kwargs)

    return {
        "method": method,
        "model": model_factory(),
        "model_factory": model_factory,
        "train_dataset": train_ds,
        "val_loader": torch.utils.data.DataLoader(train_ds, batch_size=4, collate_fn=bag_collate),
        "device": torch.device("cpu"),
        "config": {"wsi_training": {"bag_batch_size": 4}},
        "param_config": {"lr": 1e-3, "parameter": param} if param is not None else {"lr": 1e-3},
        "seed": 0,
        "is_mil": True,
        "n_classes": n_classes,
        "train_labels": train_ds.get_int_targets(),
    }


# --- grid / roster configuration -------------------------------------------------


def test_roster_for_regime_matches_report_table():
    patch = roster_for_regime(False)
    wsi = roster_for_regime(True)
    shared = {
        "ce", "balanced_sampling", "weighted_ce", "focal",
        "logit_adjustment", "post_hoc_logit_adjustment", "crt",
    }
    assert shared <= set(patch) and shared <= set(wsi)
    assert set(patch) - shared == {"ce_soft_f1", "ce_soft_mcc", "cfal", "oko"}
    assert set(wsi) - shared == {"rankmix", "sc_mil", "mde"}


def test_ce_and_crt_grids_are_lr_only():
    assert get_grid_configs("ce") == [{"lr": lr} for lr in LEARNING_RATE_GRID]
    assert get_grid_configs("crt") == [{"lr": lr} for lr in LEARNING_RATE_GRID]


def test_post_hoc_grid_has_no_learning_rate():
    configs = get_grid_configs("post_hoc_logit_adjustment")
    assert configs == [{"parameter": p} for p in GRIDS["post_hoc_logit_adjustment"]]
    assert all("lr" not in c for c in configs)


def test_weighted_ce_grid_crosses_16_configurations():
    configs = get_grid_configs("weighted_ce")
    assert len(configs) == 16
    assert {c["parameter"] for c in configs} == set(GRIDS["weighted_ce"])
    assert {c["lr"] for c in configs} == set(LEARNING_RATE_GRID)


def test_oko_grid_capped_by_k_minus_1():
    configs = get_grid_configs("oko", n_classes=3)
    assert max(c["parameter"] for c in configs) <= 2
    configs_binary = get_grid_configs("oko", n_classes=2)
    assert {c["parameter"] for c in configs_binary} == {1}


def test_update_budget_formula():
    assert update_budget(support=100, batch_size=32) == 30 * 4
    assert update_budget(support=96, batch_size=32) == 30 * 3


def test_sc_mil_batch_sampler_provides_same_class_positive_pairs():
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    sampler = ClassAwareBatchSampler(labels, batch_size=6, seed=1)
    for batch in sampler:
        counts = np.bincount(labels[batch], minlength=3)
        assert all(count == 0 or count >= 2 for count in counts)


# --- model construction ------------------------------------------------------


def test_build_model_dispatches_by_method_and_regime():
    assert isinstance(build_model("ce", False, DIM, 8, 3, 0.0), MLP)
    assert isinstance(build_model("ce", True, DIM, 8, 3, 0.0), AttentionMil)
    assert isinstance(build_model("mde", True, DIM, 8, 3, 0.0), DualExpertMil)
    assert isinstance(build_model("oko", False, DIM, 8, 3, 0.0), OkoClassifier)
    cfal = build_model("cfal", False, DIM, 8, 3, 0.0, param=2.0)
    assert isinstance(cfal, CfalPrototypeClassifier)
    assert cfal.sigma == 2.0


# --- losses --------------------------------------------------------------------


def test_mde_zero_consistency_ablation_drops_cross_term():
    model = DualExpertMil(DIM, 8, 3, 0.0)
    bags_u = [torch.randn(4, DIM), torch.randn(3, DIM)]
    bags_b = [torch.randn(4, DIM), torch.randn(3, DIM)]
    targets_u = torch.tensor([0, 1])
    targets_b = torch.tensor([1, 2])
    loss_zero = mde_bag_loss(model, bags_u, targets_u, bags_b, targets_b, lambda_con=0.0)
    loss_pos = mde_bag_loss(model, bags_u, targets_u, bags_b, targets_b, lambda_con=0.5)
    assert loss_zero.item() >= 0.0
    assert not torch.isclose(loss_zero, loss_pos)


def test_rankmix_bag_loss_finite_and_differentiable():
    student = AttentionMil(DIM, 8, 3, 0.0)
    teacher = AttentionMil(DIM, 8, 3, 0.0)
    for p in teacher.parameters():
        p.requires_grad_(False)
    bags = [torch.randn(4, DIM), torch.randn(4, DIM), torch.randn(4, DIM)]
    targets = torch.tensor([0, 1, 2])
    loss, mixed = rankmix_bag_loss(student, teacher, bags, targets, alpha=1.0)
    assert mixed == len(bags)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in student.parameters())


def test_oko_set_sampling_respects_class_membership():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    class_index = build_class_index(labels)
    rng = np.random.default_rng(0)
    pair_classes, set_indices, odd_classes = sample_oko_sets(
        class_index, n_classes=3, n_sets=20, k=1, rng=rng
    )
    assert set_indices.shape == (20, 3)
    for row, pair_cls, odd_cls in zip(set_indices, pair_classes, odd_classes, strict=True):
        assert row[0] != row[1]
        assert labels[row[0]] == pair_cls
        assert labels[row[1]] == pair_cls
        assert labels[row[2]] == odd_cls
        assert odd_cls != pair_cls


def test_oko_pair_examples_come_from_distinct_independent_units():
    """OKO's two same-class positions must be two distinct patients/slides, not two patches."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Both class-0 patches 0,1 share unit u0; patches 2,3 share u1 (two units total).
    units = np.array(["u0", "u0", "u1", "u1", "v0", "v0", "v1", "v1"])
    class_index = build_class_index(labels)
    rng = np.random.default_rng(0)

    _, set_indices, _ = sample_oko_sets(
        class_index, n_classes=2, n_sets=40, k=1, rng=rng, units=units
    )

    for row in set_indices:
        assert units[row[0]] != units[row[1]]


def test_oko_rejects_a_pair_class_with_a_single_independent_unit():
    labels = np.array([0, 0, 1, 1])
    units = np.array(["u0", "u0", "v0", "v1"])  # class 0 has only one unit
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="distinct same-class independent units"):
        sample_oko_sets(build_class_index(labels), 2, 10, 1, rng, units=units)


def test_supervised_contrastive_loss_reports_pairs_and_anchors():
    from imbalance_benchmark.modeling.losses import supervised_contrastive_loss

    embeddings = torch.randn(4, 8)
    targets = torch.tensor([0, 0, 1, 1])  # four directed positive pairs, four valid anchors

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
    loss = oko_set_loss(model, features, batch_n=5, set_size=3, pair_labels=pair_labels, odd_labels=odd_labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.main_head.weight.grad is not None
    assert model.odd_head.weight.grad is not None


# --- one-step finite training per roster method --------------------------------


@pytest.mark.parametrize(
    ("method", "param"),
    [
        ("ce", None),
        ("weighted_ce", 0.5),
        ("focal", 1.0),
        ("balanced_sampling", 0.5),
        ("logit_adjustment", 1.0),
        ("ce_soft_f1", 1.0),
        ("ce_soft_mcc", 1.0),
        ("cfal", 1.0),
        ("oko", 1),
    ],
)
def test_patch_method_one_step_finite_training(tmp_path, method, param):
    ctx = _patch_ctx(method, tmp_path, param=param)
    state, acc = fit_method({**ctx, "config": {"patch_training": {"batch_size": 8}}})
    assert 0.0 <= acc <= 1.0
    assert state


@pytest.mark.parametrize(
    ("method", "param"),
    [
        ("ce", None),
        ("weighted_ce", 0.5),
        ("focal", 1.0),
        ("balanced_sampling", 0.5),
        ("logit_adjustment", 1.0),
        ("sc_mil", 0.1),
        ("mde", 0.25),
        ("rankmix", 1.0),
    ],
)
def test_wsi_method_one_step_finite_training(tmp_path, method, param):
    ctx = _bag_ctx(method, tmp_path, param=param)
    state, acc = fit_method(ctx)
    assert 0.0 <= acc <= 1.0
    assert state


def test_fit_crt_freezes_representation_and_reinits_classifier(tmp_path):
    ctx = _patch_ctx("crt", tmp_path, param=None)
    ctx["stage_one_config"] = {"lr": 1e-3}
    ctx["param_config"] = {"lr": 1e-3}
    built_models = []
    factory = ctx["model_factory"]

    def tracking_factory():
        model = factory()
        built_models.append(model)
        return model

    ctx["model_factory"] = tracking_factory
    state, acc = fit_crt(ctx)
    assert 0.0 <= acc <= 1.0
    assert len(built_models) == 1
    stage_one_model = built_models[0]
    assert not stage_one_model.net[0].weight.requires_grad
    assert stage_one_model.net[-1].weight.requires_grad
