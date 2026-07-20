from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.datasets import features
from imbalance_benchmark.modeling.models import AttentionMil
from imbalance_benchmark.modeling.training import _fit_step


def _bootstrap_context() -> BootstrapContext:
    context = object.__new__(BootstrapContext)
    context.row_weights = np.ones((4, 3), dtype=np.int64)
    context.n_replicates = 3
    context._seed = 7
    context._seed_indices = {}
    return context


def test_secondary_endpoint_distributions_cover_classwise_and_calibration() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([labels, labels])
    probabilities = np.stack([np.eye(2)[labels], np.eye(2)[labels]])

    distributions = _bootstrap_context().secondary_distributions(
        labels,
        predictions,
        probabilities,
        ["head", "tail"],
        {"head": "head", "tail": "tail"},
    )

    assert {
        "macro_f1",
        "negative_log_likelihood",
        "macro_nll",
        "brier_score",
        "expected_calibration_error",
        "recall:head",
        "f1:tail",
        "nll:tail",
        "brier:head",
        "tier_recall:tail",
        "tier_nll:tail",
        "tier_brier:tail",
    } <= set(distributions)
    assert all(len(values) == 3 for values in distributions.values())


def test_secondary_endpoint_rows_retain_effect_estimates_and_intervals() -> None:
    from imbalance_benchmark.analysis.reporting.secondary_intervals.report import (
        _endpoint_row,
    )

    row = _endpoint_row(
        ("native", "severe", "focal"),
        "macro_f1",
        np.array([0.8, 0.7, 0.9]),
        ("native", "severe", "ce"),
        {"macro_f1": np.array([0.6, 0.5, 0.7])},
    )

    assert row["reference"] == "native/severe/ce"
    assert row["effect"] == pytest.approx(0.2)
    assert row["effect_ci_low"] == pytest.approx(0.2)
    assert row["effect_ci_high"] == pytest.approx(0.2)


def test_cost_comparisons_report_paired_effect_intervals() -> None:
    from imbalance_benchmark.analysis.reporting.secondary_intervals.costs import (
        cost_comparison_rows,
    )

    rows = []
    for split in range(3):
        for seed, ce_cost, focal_cost in ((0, 3.0, 4.0), (1, 5.0, 8.0)):
            rows.extend(
                [
                    {
                        "patient_split": split,
                        "assignment": "native",
                        "condition": "severe",
                        "method": "ce",
                        "seed_index": seed,
                        "wall_clock_seconds": ce_cost,
                    },
                    {
                        "patient_split": split,
                        "assignment": "native",
                        "condition": "severe",
                        "method": "focal",
                        "seed_index": seed,
                        "wall_clock_seconds": focal_cost,
                    },
                ]
            )

    result = cost_comparison_rows(pd.DataFrame(rows), 100, seed=3)

    assert len(result) == 1
    assert result[0]["effect"] == pytest.approx(2.0)
    assert result[0]["reference"] == "severe/ce"
    assert result[0]["ci_low"] <= result[0]["effect"] <= result[0]["ci_high"]


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


def test_mil_exposure_counts_processed_instances_not_only_bags() -> None:
    context = _mil_context("ce")

    _fit_step(
        ([torch.ones(2, 2), torch.ones(3, 2)], torch.tensor([0, 1])),
        context,
        step=0,
        max_steps=1,
    )

    assert context["processed_examples"] == 2
    assert context["processed_instances"] == 5


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


def test_upstream_wsi_tiles_require_auditable_realization_fields() -> None:
    from imbalance_benchmark.datasets.bracs.audit import validate_tile_manifest
    from imbalance_benchmark.datasets.panda import validate_tile_inventory

    with pytest.raises(ValueError, match="audit"):
        validate_tile_manifest(
            pd.DataFrame({"slide_id": ["s"], "image_path": ["tile.jpg"]}),
            expected_slides=1,
        )
    with pytest.raises(ValueError, match="audit"):
        validate_tile_inventory(
            pd.DataFrame({"slide_id": ["s"]}),
            {"s": pd.DataFrame({"image_path": ["tile.jpg"]})},
            expected_slides=1,
        )


def test_frozen_feature_reuse_verifies_revision_order_rows_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        features,
        "extract_slide_features",
        lambda paths, *_args, **_kwargs: torch.ones(len(paths), 2560),
    )
    frame = pd.DataFrame(
        {
            "slide_id": ["s1", "s1"],
            "patch_id": ["p0", "p1"],
            "image_path": ["p0.jpg", "p1.jpg"],
        }
    )
    root = tmp_path / "features"
    features.attach_extracted_features(frame, root)
    provenance = json.loads((root / "feature_provenance.json").read_text())
    assert provenance["encoder_revision"]
    assert provenance["weights_sha256"]

    with pytest.raises(ValueError, match="patch order"):
        features.attach_extracted_features(frame.iloc[::-1], root)

    tensor_path = root / "s1.pt"
    torch.save(torch.ones(1, 2560), tensor_path)
    with pytest.raises(ValueError, match="row count|hash"):
        features.attach_extracted_features(frame, root)
