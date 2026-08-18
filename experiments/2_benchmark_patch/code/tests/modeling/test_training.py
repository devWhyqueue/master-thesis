from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.datasets.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    bag_collate,
    patch_collate,
)
from imbalance_benchmark.modeling.models import (
    AttentionMil,
    MLP,
    build_model,
)
from imbalance_benchmark.modeling.special_methods import (
    fit_method,
)
from imbalance_benchmark.modeling.training import (
    ClassAwareBatchSampler,
    CHECKPOINT_INTERVAL,
    build_evaluation_loader,
    fit_model,
    initial_checkpoint,
    run_evaluation,
    update_budget,
)
from imbalance_benchmark.modeling.training import _fit_step
from imbalance_benchmark.modeling.evaluation import _gather_and_eval

DIM = 16


def _mil_context(method: str) -> dict[str, object]:
    return {
        "is_mil": True,
        "method": method,
        "device": torch.device("cpu"),
        "model": AttentionMil(2, 3, 2, dropout=0.0),
        "criterion": torch.nn.CrossEntropyLoss(),
        "param": 0.1,
        "method_diagnostics": {},
    }


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


def _bag_ctx(
    method: str, tmp_path: Path, param: float | None = None, n_classes: int = 3
) -> dict:
    manifest = _write_bag_manifest(tmp_path, n_classes=n_classes)
    train_ds = BagFeatureDataset(manifest)
    model_kwargs = {"input_dim": DIM, "hidden_dim": 8, "dropout": 0.0}

    def model_factory() -> torch.nn.Module:
        return build_model(
            method, True, n_classes=n_classes, param=param, **model_kwargs
        )

    return {
        "method": method,
        "model": model_factory(),
        "model_factory": model_factory,
        "train_dataset": train_ds,
        "val_loader": torch.utils.data.DataLoader(
            train_ds, batch_size=4, collate_fn=bag_collate
        ),
        "device": torch.device("cpu"),
        "config": {"wsi_training": {"bag_batch_size": 4}},
        "param_config": {"lr": 1e-3, "parameter": param}
        if param is not None
        else {"lr": 1e-3},
        "seed": 0,
        "is_mil": True,
        "n_classes": n_classes,
        "train_labels": train_ds.get_int_targets(),
    }


def test_sc_mil_logs_all_required_batch_diagnostics() -> None:
    context = _mil_context("sc_mil")

    _fit_step(
        ([torch.ones(2, 2) for _ in range(4)], torch.tensor([0, 0, 1, 1])),
        context,
        step=0,
        max_steps=1,
    )

    batches = context["method_diagnostics"]["sc_mil_batch_diagnostics"]
    assert batches == [
        {
            "valid_anchors": 4,
            "ordered_positive_pairs": 4,
            "represented_classes": 2,
        }
    ]


def test_update_budget_formula():
    assert update_budget(support=100, batch_size=32) == 30 * 4
    assert update_budget(support=96, batch_size=32) == 30 * 3


def test_sc_mil_batch_sampler_provides_same_class_positive_pairs():
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    sampler = ClassAwareBatchSampler(labels, batch_size=6, seed=1)
    for batch in sampler:
        counts = np.bincount(labels[batch], minlength=3)
        assert all(count == 0 or count >= 2 for count in counts)


def test_training_restores_train_mode_after_validation_checkpoint(tmp_path):
    """Dropout must remain active for updates after the first validation checkpoint."""
    ctx = _patch_ctx("ce", tmp_path, n_classes=2)
    ctx["model"] = MLP(DIM, 8, 2, dropout=0.5)

    fit_model(ctx, max_steps=CHECKPOINT_INTERVAL + 1)

    assert ctx["model"].training


def test_training_uses_the_frozen_update_budget(tmp_path):
    """A frozen budget controls fitting even if the runtime formula changes."""
    ctx = _patch_ctx("ce", tmp_path, n_classes=2)
    ctx["update_budget"] = 2

    fit_model(ctx)

    assert ctx["selected_checkpoint_step"] == 2


def test_large_patch_evaluation_batches_preserve_metrics(tmp_path: Path) -> None:
    dataset = ImbalanceDataset(
        _write_patch_manifest(tmp_path, n_classes=2, per_class=3)
    )
    model = MLP(DIM, 8, 2, dropout=0.0)
    optimized = build_evaluation_loader(dataset, is_mil=False)
    reference = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=patch_collate
    )
    optimized_result = run_evaluation(model, optimized, torch.device("cpu"), False, 2)
    reference_result = run_evaluation(model, reference, torch.device("cpu"), False, 2)

    assert optimized.batch_size == 131072
    for name in ("logits", "probs", "preds", "targets"):
        np.testing.assert_allclose(optimized_result[name], reference_result[name])
    for name in ("balanced_accuracy", "macro_f1", "nll"):
        assert optimized_result[name] == pytest.approx(reference_result[name])
    optimized_checkpoint = initial_checkpoint(
        model, optimized, torch.device("cpu"), False, 2
    )
    reference_checkpoint = initial_checkpoint(
        model, reference, torch.device("cpu"), False, 2
    )
    assert optimized_checkpoint["step"] == reference_checkpoint["step"]
    for name in ("acc", "f1", "nll"):
        assert optimized_checkpoint[name] == pytest.approx(reference_checkpoint[name])
    for name, value in optimized_checkpoint["state"].items():
        torch.testing.assert_close(value, reference_checkpoint["state"][name])


def test_evaluation_transfers_concatenated_logits_once(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = ImbalanceDataset(
        _write_patch_manifest(tmp_path, n_classes=2, per_class=2)
    )
    model = MLP(DIM, 8, 2, dropout=0.0)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=patch_collate,
    )
    original_cpu = torch.Tensor.cpu
    transfers = 0

    def record_cpu(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        nonlocal transfers
        transfers += 1
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    logits, targets = _gather_and_eval(model, loader, torch.device("cpu"), False)

    assert transfers == 1
    assert logits.shape == (4, 2)
    assert targets.tolist() == [0, 0, 1, 1]


def test_oko_reports_checkpoint_progress(tmp_path: Path, caplog) -> None:
    ctx = _patch_ctx("oko", tmp_path, param=1)
    ctx["update_budget"] = 1

    with caplog.at_level("INFO"):
        fit_method(ctx)

    assert "tune: oko seed=0 step 1/1" in caplog.text


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
        ("crt", None),
        ("sc_mil", 0.1),
        ("mde", 0.25),
    ],
)
def test_wsi_method_one_step_finite_training(tmp_path, method, param):
    ctx = _bag_ctx(method, tmp_path, param=param)
    if method == "crt":
        ctx["stage_one_config"] = {"lr": 1e-3}
    state, acc = fit_method(ctx)
    assert 0.0 <= acc <= 1.0
    assert state
    assert ctx["processed_instances"] > ctx["processed_examples"]


def test_build_optimizer_is_the_single_locked_optimizer() -> None:
    """The one optimizer factory reports the same weight decay the record records."""
    from imbalance_benchmark.modeling.training.config import (
        WEIGHT_DECAY,
        build_optimizer,
    )

    opt = build_optimizer(torch.nn.Linear(2, 2).parameters(), lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["weight_decay"] == WEIGHT_DECAY == 1e-4
    assert opt.defaults["lr"] == 1e-3


def test_resolve_training_config_records_source_only_defaults() -> None:
    """The resolved config exposes the defaults the supplied YAML never states.

    Finding: "Required run provenance is incomplete" — batch size, optimizer,
    weight decay, dropout, and checkpoint cadence were source-only.
    """
    from imbalance_benchmark.modeling.training.config import (
        TARGET_CHECKPOINTS,
        resolve_training_config,
    )

    patch = resolve_training_config({}, is_mil=False)
    assert patch["optimizer"] == "AdamW"
    assert patch["weight_decay"] == 1e-4
    assert patch["batch_size"] == 128
    assert patch["target_checkpoints"] == TARGET_CHECKPOINTS == 170
    assert patch["dropout"] == 0.1
    assert resolve_training_config({}, is_mil=True)["batch_size"] == 32


def test_resolve_checkpoint_interval_scales_with_budget() -> None:
    """Cadence targets ~TARGET_CHECKPOINTS passes; a coarser configured value still wins."""
    from imbalance_benchmark.modeling.training.config import (
        resolve_checkpoint_interval,
    )

    assert resolve_checkpoint_interval({}, False, budget=8_490) == 50
    assert resolve_checkpoint_interval({}, False, budget=523_830) == 3_082
    cfg = {"patch_training": {"checkpoint_interval": 1500}}
    assert resolve_checkpoint_interval(cfg, False, budget=257_790) == 1517
