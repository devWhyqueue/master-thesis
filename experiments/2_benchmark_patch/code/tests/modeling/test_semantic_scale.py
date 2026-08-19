from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from imbalance_benchmark.datasets.data import ImbalanceDataset
from imbalance_benchmark.datasets.features.cache import reset_feature_bank
from imbalance_benchmark.modeling.models import build_model
from imbalance_benchmark.modeling.training import _fit_step
from imbalance_benchmark.modeling.training import semantic_scale
from imbalance_benchmark.modeling.training.semantic_scale import (
    _log2_volume,
    _matched_draw_indices,
    prepare_ssb_pool,
)

DIM = 6
HIDDEN = 4


def _write_manifest(
    tmp_path: Path, classes: list[str], cases_per_class: int, patches_per_case: int
) -> Path:
    reset_feature_bank()
    rows = []
    for cls in classes:
        for case in range(cases_per_class):
            for patch in range(patches_per_case):
                slide_id = f"{cls}_S{case}_{patch}"
                feature_path = tmp_path / f"{slide_id}.pt"
                torch.save(torch.randn(1, DIM), feature_path)
                rows.append(
                    {
                        "case_id": f"{cls}_P{case}",
                        "slide_id": slide_id,
                        "cancer_type": cls,
                        "feature_path": str(feature_path),
                    }
                )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _ctx(tmp_path: Path, classes: list[str], param: float = 0.5) -> dict:
    manifest = _write_manifest(tmp_path, classes, cases_per_class=6, patches_per_case=3)
    train_ds = ImbalanceDataset(manifest, class_names=classes)
    model = build_model(
        "semantic_scale_ce",
        False,
        input_dim=DIM,
        hidden_dim=HIDDEN,
        n_classes=len(classes),
        dropout=0.0,
    )
    return {
        "method": "semantic_scale_ce",
        "model": model,
        "criterion": torch.nn.CrossEntropyLoss(),
        "train_dataset": train_ds,
        "device": torch.device("cpu"),
        "is_mil": False,
        "n_classes": len(classes),
        "seed": 0,
        "param": param,
        "processed_examples": 0,
    }


def test_matched_draw_equalizes_cases_and_patches_per_class(tmp_path: Path) -> None:
    classes = ["a", "b", "c"]
    manifest = _write_manifest(tmp_path, classes, cases_per_class=6, patches_per_case=3)
    train_ds = ImbalanceDataset(manifest, class_names=classes)

    indices, class_ids = _matched_draw_indices(train_ds, seed=0)
    counts = np.bincount(class_ids)
    assert len(set(counts.tolist())) == 1  # every class contributes equally
    for class_id, name in enumerate(classes):
        rows = train_ds.df.loc[indices[class_ids == class_id]]
        assert rows["cancer_type"].eq(name).all()
        assert rows["case_id"].nunique() <= 6


def test_log2_volume_grows_with_feature_spread() -> None:
    torch.manual_seed(0)
    concentrated = torch.randn(20, 8) * 0.01
    spread = torch.randn(20, 8) * 5.0
    v_concentrated = _log2_volume(concentrated - concentrated.mean(0), d=8)
    v_spread = _log2_volume(spread - spread.mean(0), d=8)
    assert v_spread > v_concentrated


def test_ssb_fit_step_runs_all_three_stages_without_crashing(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, ["a", "b", "c"])
    b_size = 4
    prepare_ssb_pool(ctx, b_size)
    updates_per_pass = ctx["ssb_updates_per_pass"]
    max_steps = 7 * updates_per_pass  # run past pass 6 into reweighted steps

    batch = {
        "features": torch.randn(b_size, DIM),
        "target": torch.randint(0, 3, (b_size,)),
    }
    for step in range(max_steps):
        loss = _fit_step(batch, ctx, step, max_steps)
        assert torch.isfinite(loss)

    pool = ctx["ssb_pool"]
    assert pool.volumes  # some class earned a real estimate by the end
    assert pool.filled is not None and bool(pool.filled.all())  # full traversal


def test_ssb_uses_unit_weights_for_exactly_five_passes(tmp_path: Path) -> None:
    torch.manual_seed(0)
    ctx = _ctx(tmp_path, ["a", "b", "c"])
    b_size = 4
    prepare_ssb_pool(ctx, b_size)
    updates_per_pass = ctx["ssb_updates_per_pass"]
    max_steps = 7 * updates_per_pass

    batch = {
        "features": torch.randn(b_size, DIM),
        "target": torch.randint(0, 3, (b_size,)),
    }
    reference = torch.nn.functional.cross_entropy(
        ctx["model"](batch["features"]), batch["target"]
    )
    for step in range(5 * updates_per_pass):
        loss = _fit_step(batch, ctx, step, max_steps)
        assert torch.allclose(loss, reference)  # unit weights: plain CE

    weighted_loss = _fit_step(batch, ctx, 5 * updates_per_pass, max_steps)
    assert not torch.allclose(weighted_loss, reference)  # SSB weights from here on
    assert ctx["ssb_pool"].volumes  # computed ahead of the first weighted step


def test_update_volumes_runs_once_per_pool_pass_not_per_step(
    tmp_path: Path, monkeypatch
) -> None:
    ctx = _ctx(tmp_path, ["a", "b", "c"])
    b_size = 4
    prepare_ssb_pool(ctx, b_size)
    updates_per_pass = ctx["ssb_updates_per_pass"]
    max_steps = 7 * updates_per_pass  # run past pass 6

    calls = 0
    original = semantic_scale._update_volumes

    def counting_update_volumes(pool, n_classes):
        nonlocal calls
        calls += 1
        return original(pool, n_classes)

    monkeypatch.setattr(semantic_scale, "_update_volumes", counting_update_volumes)

    batch = {
        "features": torch.randn(b_size, DIM),
        "target": torch.randint(0, 3, (b_size,)),
    }
    for step in range(max_steps):
        semantic_scale.ssb_loss(ctx["model"], batch["features"], batch["target"], ctx, step)

    # one seeding call at reweight_step - 1, plus one per completed pool pass
    # thereafter (passes 6 and 7) -- never one per step.
    assert calls == 3


def test_ssb_pool_class_names_match_dataset_order(tmp_path: Path) -> None:
    classes = ["a", "b", "c"]
    ctx = _ctx(tmp_path, classes)
    prepare_ssb_pool(ctx, 4)
    pool = ctx["ssb_pool"]
    assert set(pool.class_ids.tolist()) == {0, 1, 2}
