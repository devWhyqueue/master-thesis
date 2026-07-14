from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.analysis.predictors import rq3_analysis
from imbalance_benchmark.analysis.reporting import tables
from imbalance_benchmark.commands import confirm, tuning
from imbalance_benchmark.commands import prepare
from imbalance_benchmark.commands.pilot import _pilot_setup
from imbalance_benchmark.datasets.bracs import LABELS as BRACS_LABELS
from imbalance_benchmark.datasets import features
from imbalance_benchmark.modeling.context import Regime
from imbalance_benchmark.modeling.workflows.confirmation import RunContext


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


def test_pilot_uses_the_same_semantic_bracs_order_as_the_freeze(tmp_path: Path) -> None:
    rows = [
        {
            "case_id": f"patient-{name}",
            "slide_id": f"slide-{name}",
            "cancer_type": name,
            "split": "train",
        }
        for name in BRACS_LABELS
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)

    _, class_names, *_ = _pilot_setup({"data": tmp_path}, {"dataset": {}})

    assert class_names == list(BRACS_LABELS)


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

    monkeypatch.setattr(tuning, "load_training_dataset", load_dataset)
    tuning._combined_scopes(
        [({"data": tmp_path}, regime, object())], "moderate", ("native",)
    )

    assert observed == [locked]


def test_rq3_effective_support_uses_condition_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    balanced = tmp_path / "manifest_balanced.csv"
    condition = tmp_path / "manifest_native_severe.csv"
    validation = tmp_path / "manifest.csv"
    for path in (balanced, condition, validation):
        path.write_text("placeholder", encoding="utf-8")
    reference_x = np.array([[1.0], [2.0], [3.0], [4.0]])
    reference_y = np.array([0, 0, 1, 1])
    condition_x = np.array([[1.0], [4.0]])
    condition_y = np.array([0, 1])

    def feature_frame(path: Path, *_: object) -> tuple[np.ndarray, np.ndarray]:
        return (
            (condition_x, condition_y)
            if path == condition
            else (reference_x, reference_y)
        )

    def identity(path: Path, *_: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "case_id": ["p0", "p0", "p1", "p1"]
                if path == balanced
                else ["p0", "p1"],
                "slide_id": ["s0", "s1", "s2", "s3"]
                if path == balanced
                else ["s0", "s3"],
            }
        )

    monkeypatch.setattr(rq3_analysis, "_feature_frame", feature_frame)
    monkeypatch.setattr(rq3_analysis, "_feature_identity", identity)
    monkeypatch.setattr(
        rq3_analysis,
        "intrinsic_separability",
        lambda *_: {
            "linear_probe_macro_recall": 0.5,
            "knn_macro_recall": 0.5,
            "per_class_nn_error": {},
        },
    )
    monkeypatch.setattr(
        rq3_analysis,
        "condition_learnability",
        lambda *_: {"linear_probe_macro_recall": 0.5},
    )
    monkeypatch.setattr(rq3_analysis, "class_margin_cross_fit", lambda x, *_: x[:, 0])
    monkeypatch.setattr(rq3_analysis, "intraclass_correlation", lambda *_: 0.0)

    result = rq3_analysis._covariates(
        {"data": tmp_path},
        False,
        {"path": str(condition), "contribution_stats": {}},
        {"class_names": ["A", "B"]},
    )

    assert result["log_effective_support"] == pytest.approx(0.0)


def test_rq3_wsi_records_patient_support_without_patch_effective_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """WSI RQ3 uses slide support plus patient support, never patch sensitivity."""
    balanced = tmp_path / "manifest_balanced.csv"
    condition = tmp_path / "manifest_native_severe.csv"
    validation = tmp_path / "manifest.csv"
    for path in (balanced, condition, validation):
        path.write_text("placeholder", encoding="utf-8")
    features = np.array([[1.0], [2.0], [3.0], [4.0]])
    labels = np.array([0, 0, 1, 1])
    monkeypatch.setattr(rq3_analysis, "_feature_frame", lambda *_: (features, labels))
    monkeypatch.setattr(
        rq3_analysis,
        "intrinsic_separability",
        lambda *_: {
            "linear_probe_macro_recall": 0.5,
            "knn_macro_recall": 0.5,
            "per_class_nn_error": {},
        },
    )
    monkeypatch.setattr(
        rq3_analysis,
        "condition_learnability",
        lambda *_: {"linear_probe_macro_recall": 0.5},
    )
    monkeypatch.setattr(
        rq3_analysis,
        "_feature_identity",
        lambda *_: (_ for _ in ()).throw(AssertionError("WSI must not compute N_eff")),
    )

    result = rq3_analysis._covariates(
        {"data": tmp_path},
        True,
        {
            "path": str(condition),
            "contribution_stats": {
                "A": {"n_slides": 4, "n_patients": 2},
                "B": {"n_slides": 6, "n_patients": 3},
            },
        },
        {"class_names": ["A", "B"]},
    )

    assert result["log_min_support"] == pytest.approx(np.log(4))
    assert result["log_min_patient_support"] == pytest.approx(np.log(2))
    assert "log_effective_support" not in result


def test_tuning_rejects_a_single_split_as_a_definitive_selection() -> None:
    args = argparse.Namespace(split_index=0, config=None, seed=0)

    with pytest.raises(ValueError, match="all three"):
        tuning.cmd_tune(args)


def test_feature_extraction_rejects_a_non_virchow2_encoder() -> None:
    with pytest.raises(ValueError, match="Virchow2"):
        features.resolve_feature_provenance({"model_name": "resnet50"})


def test_prepare_validates_encoder_for_precomputed_feature_manifests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        prepare,
        "build_manifest",
        lambda _: pd.DataFrame({"feature_path": ["cached.pt"]}),
    )

    with pytest.raises(ValueError, match="Virchow2"):
        prepare._base_manifest(
            {
                "dataset": {"name": "precomputed"},
                "feature_extraction": {"model_name": "resnet50"},
            },
            {"data": tmp_path},
        )


def test_feature_cache_rejects_metadata_from_a_different_encoder_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame({"slide_id": ["s1"], "image_path": ["s1.jpg"]})
    monkeypatch.setattr(
        features,
        "extract_slide_features",
        lambda *_args, **_kwargs: torch.ones(1, 2560),
    )
    root = tmp_path / "features"
    features.attach_extracted_features(frame, root, dtype="float16")

    with pytest.raises(ValueError, match="provenance"):
        features.attach_extracted_features(frame, root, dtype="float32")
