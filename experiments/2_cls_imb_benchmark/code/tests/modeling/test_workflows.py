from __future__ import annotations

from pathlib import Path

import pytest
import torch

from imbalance_benchmark.analysis.query import load_seed_predictions
from imbalance_benchmark.commands import confirm, tuning
from imbalance_benchmark.common import write_run_record
from imbalance_benchmark.datasets.bracs import LABELS as BRACS_LABELS
from imbalance_benchmark.modeling.context import (
    GRIDS,
    LEARNING_RATE_GRID,
    get_grid_configs,
    roster_for_regime,
)
from imbalance_benchmark.modeling.context import Regime
from imbalance_benchmark.modeling.workflows.confirmation import (
    RunContext,
    confirm_method,
)

def _write_seed_record(method_dir: Path, seed_idx: int) -> None:
    write_run_record(
        method_dir / f"seed={seed_idx}",
        {
            "benchmark": "patch",
            "class_names": ["A", "B"],
            "splits": {
                "test": {
                    "labels": [0, 1],
                    "preds": [0, 1],
                    "probabilities": [[0.9, 0.1], [0.2, 0.8]],
                    "logits": [[2.0, 0.0], [0.0, 2.0]],
                }
            },
        },
    )

def test_train_time_logit_adjustment_confirmation_preserves_selected_tau(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Confirmation must evaluate train-time logit adjustment with its tuned tau."""
    observed: list[float] = []
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation.build_training_ctx",
        lambda *_: {"seed": 7},
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._timed_fit",
        lambda *_: ({}, 0.0),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._run_and_record",
        lambda *args: observed.append(float(args[7])),
    )
    run = RunContext(
        device=torch.device("cpu"),
        config={},
        n_classes=2,
        is_mil=False,
        class_names=["A", "B"],
        val_loader=object(),
        test_loader=object(),
        paths={"data": tmp_path, "results": tmp_path},
        seeds=[7],
        assignment="native",
    )

    confirm_method("severe", "logit_adjustment", {"parameter": 0.25}, object(), run)

    assert observed == [0.25]

def test_confirmation_condition_uses_the_frozen_class_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    locked = list(BRACS_LABELS)
    observed: list[list[str] | None] = []

    def load_dataset(*_: object, **kwargs: object) -> object:
        observed.append(kwargs.get("class_names"))
        return object()

    monkeypatch.setattr(confirm, "load_training_dataset", load_dataset)
    monkeypatch.setattr(confirm, "confirm_ce", lambda *args: [])
    run = RunContext(
        device=torch.device("cpu"),
        config={},
        n_classes=len(locked),
        is_mil=False,
        class_names=locked,
        val_loader=object(),
        test_loader=object(),
        paths={"data": tmp_path},
        seeds=[],
        assignment="native",
    )

    confirm._confirm_condition("moderate", ("ce",), {"ce": {}}, run)

    assert observed == [locked]

def test_combined_tuning_scope_passes_the_frozen_class_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    locked = list(BRACS_LABELS)
    regime = Regime(
        torch.device("cpu"), {}, len(locked), False, locked_class_names=locked
    )
    observed: list[list[str] | None] = []

    def load_dataset(*_: object, **kwargs: object) -> object:
        observed.append(kwargs.get("class_names"))
        return object()

    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.tuning.tuning_shards.load_training_dataset",
        load_dataset,
    )
    tuning.combined_scopes(
        [({"data": tmp_path}, regime, object())], "moderate", ("native",)
    )

    assert observed == [locked]

def test_roster_for_regime_matches_report_table():
    patch = roster_for_regime(False)
    wsi = roster_for_regime(True)
    shared = {
        "ce",
        "balanced_sampling",
        "weighted_ce",
        "focal",
        "logit_adjustment",
        "post_hoc_logit_adjustment",
        "crt",
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

def test_confirmation_provenance_payload_carries_appendix_a_fields() -> None:
    """The run record's provenance carries the Appendix A fields flagged as missing.

    Finding: "Required run provenance is incomplete" — model/optimizer config,
    candidate grid, freeze hash, dataset version, achieved T/rho, and pilot quota.
    """
    from imbalance_benchmark.modeling.workflows.confirmation_helpers import (
        RunContext,
        _provenance_payload,
    )

    run = RunContext(
        device=torch.device("cpu"),
        config={},
        n_classes=2,
        is_mil=False,
        val_loader=None,  # type: ignore[arg-type]
        test_loader=None,  # type: ignore[arg-type]
        paths={},
        seeds=[0],
        class_names=["A", "B"],
        assignment="native",
    )
    freeze = {
        "content_sha256": "freezehash",
        "dataset_provenance": {"version": "v1"},
        "shared_T": 100,
        "min_support": 20,
        "method_grids": {"weighted_ce": [{"lr": 1e-4, "parameter": 1.0}]},
        "assignment_conditions": {"native": {"severe": {"achieved_rho": 99.0}}},
    }

    out = _provenance_payload(run, "severe", "weighted_ce", freeze)
    assert out["model_optimizer_config"]["optimizer"] == "AdamW"
    assert out["model_optimizer_config"]["weight_decay"] == 1e-4
    assert out["candidate_grid"] == [{"lr": 1e-4, "parameter": 1.0}]
    assert out["freeze_content_sha256"] == "freezehash"
    assert out["dataset_version"] == "v1"
    assert out["achieved_T"] == 100
    assert out["achieved_rho"] == 99.0
    assert out["pilot_min_support"] == 20

def test_partial_confirmation_block_is_rejected(tmp_path: Path) -> None:
    """Fewer than five valid confirmation seeds must stop inference, not be averaged."""
    paths = {"results": tmp_path}
    method_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce"
    for seed_idx in range(3):  # only three of the required five present
        _write_seed_record(method_dir, seed_idx)

    with pytest.raises(RuntimeError, match="incomplete"):
        load_seed_predictions(paths, "severe", "weighted_ce", "native")

def test_complete_confirmation_block_stacks_all_five_seeds(tmp_path: Path) -> None:
    paths = {"results": tmp_path}
    method_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce"
    for seed_idx in range(5):
        _write_seed_record(method_dir, seed_idx)

    stacked = load_seed_predictions(paths, "severe", "weighted_ce", "native")

    assert stacked is not None
    assert stacked["preds"].shape[0] == 5

def test_missing_confirmation_method_is_not_silently_skipped(tmp_path: Path) -> None:
    """A roster method with no directory is a failed confirmation block."""
    with pytest.raises(RuntimeError, match="missing"):
        load_seed_predictions({"results": tmp_path}, "severe", "weighted_ce", "native")
