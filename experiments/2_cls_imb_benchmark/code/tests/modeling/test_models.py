from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.datasets.data import (
    ImbalanceDataset,
    patch_collate,
)
from imbalance_benchmark.modeling.context import RunExposure, cost_payload
from imbalance_benchmark.modeling.context import set_training_mode
from imbalance_benchmark.modeling.models import (
    AttentionMil,
    CfalPrototypeClassifier,
    DualExpertMil,
    MLP,
    OkoClassifier,
    build_model,
)
from imbalance_benchmark.modeling.oko import (
    build_class_index,
    sample_oko_sets,
)
from imbalance_benchmark.modeling.special_methods import (
    fit_crt,
    mde_bag_loss,
)
from imbalance_benchmark.modeling.training import FIXED_BALANCED_SAMPLER_METHODS
from imbalance_benchmark.modeling.workflows.multistage import (
    _freeze_and_reinit_classifier,
)

DIM = 16

def _write_patch_manifest(
    tmp_path: Path, n_classes: int = 3, per_class: int = 8
) -> Path:
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

def _patch_ctx(
    method: str, tmp_path: Path, param: float | None = None, n_classes: int = 3
) -> dict:
    manifest = _write_patch_manifest(tmp_path, n_classes=n_classes)
    train_ds = ImbalanceDataset(manifest)
    model_kwargs = {"input_dim": DIM, "hidden_dim": 8, "dropout": 0.0}

    def model_factory() -> torch.nn.Module:
        return build_model(
            method, False, n_classes=n_classes, param=param, **model_kwargs
        )

    return {
        "method": method,
        "model": model_factory(),
        "model_factory": model_factory,
        "train_dataset": train_ds,
        "val_loader": torch.utils.data.DataLoader(
            train_ds, batch_size=8, collate_fn=patch_collate
        ),
        "device": torch.device("cpu"),
        "config": {"patch_training": {"batch_size": 8}},
        "param_config": {"lr": 1e-3, "parameter": param}
        if param is not None
        else {"lr": 1e-3},
        "seed": 0,
        "is_mil": False,
        "n_classes": n_classes,
        "train_labels": train_ds.get_int_targets(),
    }

def test_post_hoc_cost_has_no_trainable_network_parameters() -> None:
    """A reused checkpoint is evaluated but never updated by post-hoc adjustment."""
    cost = cost_payload(
        "post_hoc_logit_adjustment",
        3,
        0.0,
        torch.nn.Linear(2, 2),
        RunExposure(4, 4, 0),
    )

    assert cost["trainable_parameters"] == 0

def test_rankmix_cost_records_teacher_and_student_training_footprint() -> None:
    """RankMix trains two networks, not only the final student checkpoint."""
    cost = cost_payload(
        "rankmix",
        3,
        1.0,
        torch.nn.Linear(2, 2),
        RunExposure(4, 4, 8, training_footprint_parameters=12),
    )

    assert cost["training_footprint_parameters"] == 12

@pytest.mark.parametrize("is_mil", [False, True])
def test_crt_freezes_the_full_representation_in_deterministic_eval_mode(
    is_mil: bool,
) -> None:
    model = (
        AttentionMil(input_dim=4, hidden_dim=3, output_dim=2, dropout=0.5)
        if is_mil
        else MLP(input_dim=4, hidden_dim=3, output_dim=2, dropout=0.5)
    )

    frozen = _freeze_and_reinit_classifier(model, is_mil)
    set_training_mode({"model": model, "frozen_eval_modules": frozen})

    assert all(
        not parameter.requires_grad
        for module in frozen
        for parameter in module.parameters()
    )
    assert all(not module.training for module in frozen)
    assert (
        all(parameter.requires_grad for parameter in model.classifier.parameters())
        if is_mil
        else all(parameter.requires_grad for parameter in model.net[-1].parameters())
    )

def test_build_model_dispatches_by_method_and_regime():
    assert isinstance(build_model("ce", False, DIM, 8, 3, 0.0), MLP)
    assert isinstance(build_model("ce", True, DIM, 8, 3, 0.0), AttentionMil)
    assert isinstance(build_model("mde", True, DIM, 8, 3, 0.0), DualExpertMil)
    assert isinstance(build_model("oko", False, DIM, 8, 3, 0.0), OkoClassifier)
    cfal = build_model("cfal", False, DIM, 8, 3, 0.0, param=2.0)
    assert isinstance(cfal, CfalPrototypeClassifier)
    assert cfal.sigma == 2.0

def test_mde_zero_consistency_ablation_drops_cross_term():
    model = DualExpertMil(DIM, 8, 3, 0.0)
    bags_u = [torch.randn(4, DIM), torch.randn(3, DIM)]
    bags_b = [torch.randn(4, DIM), torch.randn(3, DIM)]
    targets_u = torch.tensor([0, 1])
    targets_b = torch.tensor([1, 2])
    loss_zero = mde_bag_loss(
        model, bags_u, targets_u, bags_b, targets_b, lambda_con=0.0
    )
    loss_pos = mde_bag_loss(model, bags_u, targets_u, bags_b, targets_b, lambda_con=0.5)
    assert loss_zero.item() >= 0.0
    assert not torch.isclose(loss_zero, loss_pos)

def test_oko_set_sampling_respects_class_membership():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    class_index = build_class_index(labels)
    rng = np.random.default_rng(0)
    pair_classes, set_indices, odd_classes = sample_oko_sets(
        class_index, n_classes=3, n_sets=20, k=1, rng=rng
    )
    assert set_indices.shape == (20, 3)
    for row, pair_cls, odd_cls in zip(
        set_indices, pair_classes, odd_classes, strict=True
    ):
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
    # cRT must reuse the already-seeded context model for stage one rather
    # than creating a second, differently initialized model.
    assert len(built_models) == 0
    stage_one_model = ctx["model"]
    assert not stage_one_model.net[0].weight.requires_grad
    assert stage_one_model.net[-1].weight.requires_grad

def test_scholz_methods_are_the_balanced_sampler_hybrids():
    assert "ce_soft_f1" in FIXED_BALANCED_SAMPLER_METHODS
    assert "ce_soft_mcc" in FIXED_BALANCED_SAMPLER_METHODS
    assert "rankmix" not in FIXED_BALANCED_SAMPLER_METHODS

def test_oko_odd_classes_are_distinct_and_exclude_the_pair_class():
    n_classes, k = 5, 3
    class_index = {c: [2 * c, 2 * c + 1] for c in range(n_classes)}  # idx // 2 == class
    pair, sets, _ = sample_oko_sets(class_index, n_classes, 64, k, np.random.default_rng(0))
    odd_classes = sets[:, 2:] // 2
    for row, pair_class in zip(odd_classes.tolist(), pair.tolist()):
        assert len(set(row)) == k
        assert pair_class not in row
