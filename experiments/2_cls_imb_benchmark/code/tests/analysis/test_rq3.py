from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.analysis.inference.gates import (
    _SeverityInputs,
    _recovery_comparison,
    discrimination_gate_comparison,
)
from imbalance_benchmark.analysis.inference.gates import (
    deficit,
    recovery,
)
from imbalance_benchmark.analysis.metrics import (
    classification_payload,
)
from imbalance_benchmark.analysis.predictors import rq3_analysis
from imbalance_benchmark.analysis.predictors.hierarchical_models import _log_scale_prior
from imbalance_benchmark.analysis.predictors.rq3_analysis import (
    _cells,
    _covariates,
)
from imbalance_benchmark.analysis.predictors.rq3_cross_split import _comparison_maps
from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    fit_deficit_model,
    fit_gate_pass_model,
    fit_recovery_model,
)
from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    fit_linked_sensitivity_models,
)
from imbalance_benchmark.common import (
    write_json,
    write_run_record,
)

def _toy_identity(n_patients_per_class: int = 6) -> pd.DataFrame:
    rows = []
    for cls in ["A", "B"]:
        for p in range(n_patients_per_class):
            case_id = f"{cls}_P{p}"
            for s in range(2):
                rows.append(
                    {
                        "case_id": case_id,
                        "slide_id": f"{case_id}_S{s}",
                        "cancer_type": cls,
                    }
                )
    return pd.DataFrame(rows)

def _write_fake_run(
    results_root: Path,
    condition: str,
    method: str,
    seed_idx: int,
    class_names: list[str],
    seed_offset: int,
) -> None:
    rng = np.random.default_rng(seed_offset)
    n = 12
    labels = rng.integers(0, len(class_names), size=n)
    probs = rng.dirichlet(np.ones(len(class_names)), size=n)
    preds = probs.argmax(axis=1)
    payload = classification_payload(
        labels.tolist(), preds.tolist(), probs.tolist(), class_names
    )
    payload["labels"] = labels.tolist()
    payload["preds"] = preds.tolist()
    payload["probabilities"] = probs.tolist()
    payload["logits"] = np.log(np.clip(probs, 1e-6, 1.0)).tolist()
    write_run_record(
        results_root / condition / method / f"seed={seed_idx}",
        {
            "benchmark": "patch",
            "condition": condition,
            "method": method,
            "seed": seed_offset,
            "class_names": class_names,
            "tuning_params": {"lr": 1e-3},
            "cost": {"updates": 10},
            "splits": {"validation": payload, "test": payload},
        },
    )

def _rq3_cell(group: str, method: str, rho: float, deficit: float, gate: bool) -> dict:
    return {
        "group": group,
        "method": method,
        "rho": rho,
        "support_difficulty_alignment": 0.2,
        "separability": 0.5,
        "learnability": 0.4,
        "log_min_support": 3.0,
        "is_wsi": 0.0 if "patch" in group else 1.0,
        "gate_passed": gate,
        "deficit_ba": deficit,
        "deficit_se": 0.01,
        "recovery": 0.5,
        "recovery_se": 0.1,
    }

@pytest.mark.parametrize(
    "method_metric,imbalanced_ce,d,expected",
    [
        (0.5, 0.5, 0.1, 0.0),
        (0.6, 0.5, 0.1, 1.0),
        (0.4, 0.5, 0.1, -1.0),
        (0.7, 0.5, 0.1, 2.0),
    ],
)
def test_recovery_sign_conventions(method_metric, imbalanced_ce, d, expected):
    assert recovery(method_metric, imbalanced_ce, d) == pytest.approx(expected)

def test_recovery_nan_when_no_deficit():
    assert np.isnan(recovery(0.6, 0.5, 0.0))

def test_deficit_higher_is_better():
    assert deficit(0.7, 0.5) == pytest.approx(0.2)

def test_rq3_gate_pass_and_deficit_and_recovery_models_run():
    rng = np.random.default_rng(0)
    groups = [f"dataset_{i % 4}" for i in range(24)]
    cells = []
    for i in range(24):
        rho = float(rng.choice([1.0, 10.0, 100.0]))
        separability = float(rng.normal())
        gate_passed = rho > 1.0
        cells.append(
            {
                "group": groups[i],
                "rho": rho,
                "support_difficulty_alignment": 0.2,
                "separability": separability,
                "gate_passed": gate_passed,
                "deficit_ba": float(rng.normal(0.05, 0.02)),
                "deficit_se": 0.01,
                "recovery": float(rng.normal(0.5, 0.1)),
                "recovery_se": 0.05,
            }
        )
    gate_model = fit_gate_pass_model(cells)
    deficit_model = fit_deficit_model(cells)
    recovery_model = fit_recovery_model(cells)
    assert "slopes" in gate_model and len(gate_model["slopes"]) == 2
    assert "slopes" in deficit_model
    assert "slopes" in recovery_model

def test_rq3_recovery_model_empty_when_no_gated_cells():
    cells = [
        {
            "group": "g",
            "rho": 1.0,
            "separability": 0.0,
            "gate_passed": False,
            "recovery": 0.0,
            "recovery_se": 0.01,
        }
    ]
    assert fit_recovery_model(cells) == {}

def test_rq3_cells_keep_calibration_gate_recovery(monkeypatch: pytest.MonkeyPatch):
    """A calibration-only cell must use tail-NLL recovery, not BA recovery."""
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._covariates",
        lambda *_: {"separability": 0.5},
    )
    comparisons = [
        {
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "discrimination",
            "gate_passed": False,
            "effect": 0.01,
            "bootstrap_effect": [0.01, 0.02],
        },
        {
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "calibration",
            "gate_passed": True,
            "effect": 0.08,
            "bootstrap_effect": [0.08, 0.09],
        },
        {
            "assignment": "native",
            "severity": "severe",
            "method": "weighted_ce",
            "gate": "calibration",
            "gate_passed": True,
            "effect": 0.04,
            "recovery": 0.5,
            "bootstrap_effect": [0.04, 0.05],
            "bootstrap_numerator": [0.04, 0.05],
            "bootstrap_denominator": [0.08, 0.10],
        },
    ]
    freeze = {
        "difficulty_evidence": {"difficulty": {"A": 0.1, "B": 0.2}},
        "assignment_conditions": {
            "native": {"severe": {"achieved_rho": 100.0, "allocated_counts": {"A": 10, "B": 100}}}
        },
    }

    cells = _cells({}, comparisons, freeze, "dataset:target", False)

    recovery = next(cell for cell in cells if cell["method"] == "weighted_ce")
    assert recovery["gate"] == "calibration"
    assert recovery["gate_passed"] is True
    assert recovery["recovery"] == pytest.approx(0.5)

def test_cross_split_rq3_keeps_gate_specific_calibration_outcome():
    rows = [
        {
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "calibration",
            "gate_passed": True,
        },
        {
            "assignment": "native",
            "severity": "severe",
            "method": "weighted_ce",
            "gate": "calibration",
            "bootstrap_numerator": [0.04],
            "bootstrap_denominator": [0.08],
        },
    ]

    gates, outcomes = _comparison_maps(rows)

    assert gates[("native", "severe", "calibration")]
    assert ("native", "severe", "weighted_ce", "calibration") in outcomes

def test_log_scale_prior_prevents_random_effect_scale_collapse():
    collapsed = _log_scale_prior(torch.tensor([-20.0]))
    centered = _log_scale_prior(torch.tensor([0.0]))
    assert collapsed > centered

def test_rq3_regime_specific_sensitivities_fit_their_eligible_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch and WSI support sensitivities retain their respective regimes."""
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_wiring.fit_rq3_model",
        lambda y, x, *_args, **_kwargs: {
            "n_observations": len(y),
            "values": x.ravel().tolist(),
        },
    )
    patch_cells = [
        {
            "group": "patch",
            "gate_passed": True,
            "deficit_ba": 0.1,
            "deficit_se": 0.01,
            "recovery": 0.5,
            "recovery_se": 0.02,
            "log_effective_support": value,
        }
        for value in (2.0, 3.0)
    ]
    wsi_cells = [
        {
            "group": "wsi",
            "gate_passed": True,
            "deficit_ba": 0.1,
            "deficit_se": 0.01,
            "recovery": 0.5,
            "recovery_se": 0.02,
            "log_min_patient_support": value,
        }
        for value in (4.0, 5.0)
    ]

    sensitivity = fit_linked_sensitivity_models(
        patch_cells + wsi_cells, patch_cells + wsi_cells
    )

    assert sensitivity["log_effective_support"]["deficit"]["values"] == [2.0, 3.0]
    assert sensitivity["log_min_patient_support"]["deficit"]["values"] == [4.0, 5.0]

def test_rq3_icc_margin_uses_the_fixed_intrinsic_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ref_x = np.array([[1.0], [2.0]])
    ref_y = np.array([0, 1])
    cond_x = np.array([[10.0], [20.0]])
    cond_y = np.array([0, 1])
    seen: dict[str, np.ndarray] = {}

    def feature_frame(path: Path, *_: object) -> tuple[np.ndarray, np.ndarray]:
        if path.name == "manifest_balanced.csv":
            return ref_x, ref_y
        if path.name == "condition.csv":
            return cond_x, cond_y
        return ref_x, ref_y

    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._feature_frame",
        feature_frame,
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._feature_identity",
        lambda path, *_: pd.DataFrame(
            {"case_id": ["a", "b"], "slide_id": ["s1", "s2"]}
        ),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.intrinsic_separability",
        lambda *_: {
            "linear_probe_macro_recall": 0.5,
            "knn_macro_recall": 0.5,
            "per_class_nn_error": {},
        },
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.condition_learnability",
        lambda *_: {"linear_probe_macro_recall": 0.5},
    )

    def margins(x: np.ndarray, *_: object) -> np.ndarray:
        seen["x"] = x
        return np.array([0.1, 0.2])

    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.class_margin_cross_fit",
        margins,
    )

    condition_path = tmp_path / "condition.csv"
    condition_path.touch()
    _covariates(
        {"data": tmp_path},
        False,
        {"path": str(condition_path), "contribution_stats": {}},
    )

    assert np.array_equal(seen["x"], ref_x)

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

def test_rq3_crossed_cell_uses_observed_point_not_bootstrap_mean() -> None:
    """RQ3 crossed cells report the observed (index-0) deficit and recovery.

    Replicate 0 is the observed cross-split estimate; replicates 1.. supply only
    the spread. The prior code averaged over every replicate, biasing the point
    estimate (finding: "RQ3 uses the wrong point estimates").
    """
    from imbalance_benchmark.analysis.predictors.rq3_cross_split import _crossed_cell

    gates = {("native", "severe", "discrimination"): True}

    ce_cell = {
        "assignment": "native",
        "severity": "severe",
        "method": "ce",
        "gate": "discrimination",
    }
    ce_row = {
        "assignment": "native",
        "severity": "severe",
        "gate": "discrimination",
        "bootstrap_effect": [0.5, 2.0],
    }
    ce_out = _crossed_cell(ce_cell, gates, ce_row)
    assert ce_out["deficit_ba"] == pytest.approx(0.5)  # observed, not mean 1.25
    assert ce_out["deficit_se"] == pytest.approx(np.std([0.5, 2.0], ddof=1))

    rec_cell = {
        "assignment": "native",
        "severity": "severe",
        "method": "weighted_ce",
        "gate": "discrimination",
    }
    rec_row = {
        "bootstrap_numerator": [1.0, 4.0],
        "bootstrap_denominator": [2.0, 2.0],
    }
    rec_out = _crossed_cell(rec_cell, gates, rec_row)
    assert rec_out["recovery"] == pytest.approx(0.5)  # observed 1/2, not mean 1.25
    assert rec_out["recovery_se"] == pytest.approx(np.std([0.5, 2.0], ddof=1))

def test_rq3_crossed_cell_treats_a_near_zero_denominator_as_undefined() -> None:
    """A tiny but nonzero deficit must not blow recovery up to an arbitrary value.

    Finding: a denominator of 1e-9 produced recovery=-6.7e7, which then
    silently entered ``load_rq3_cells``'s cross-split average. Only an exact
    zero was previously guarded; this widens the guard to a tolerance.
    """
    from imbalance_benchmark.analysis.predictors.rq3_cross_split import _crossed_cell

    gates = {("native", "severe", "discrimination"): True}
    rec_cell = {
        "assignment": "native",
        "severity": "severe",
        "method": "weighted_ce",
        "gate": "discrimination",
    }
    rec_row = {
        "bootstrap_numerator": [0.067, 4.0],
        "bootstrap_denominator": [1e-9, 2.0],
    }

    rec_out = _crossed_cell(rec_cell, gates, rec_row)

    assert np.isnan(rec_out["recovery"])
    assert np.isnan(rec_out["recovery_se"]) or rec_out["recovery_se"] == pytest.approx(
        0.0
    )

def test_cross_dataset_rq3_pools_groups_and_reports_stability() -> None:
    """RQ3's combined fit spans dataset-target groups with LODO and sensitivity fits."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import cross_dataset_rq3

    cells = []
    for i, group in enumerate(["tcga:patch", "tcga:wsi", "bracs:patch", "panda:wsi"]):
        cells.append(_rq3_cell(group, "ce", 10.0 + i, 0.05 + 0.01 * i, gate=True))
        cells.append(_rq3_cell(group, "weighted_ce", 10.0 + i, np.nan, gate=True))

    report = cross_dataset_rq3(cells)

    assert report["n_groups"] == 4
    assert len(report["models"]["deficit"]["rand_intercepts"]) == 4
    assert set(report["sensitivity"]) == {
        "separability",
        "learnability",
        "log_min_support",
        "is_wsi",
    }
    assert set(report["leave_one_group_out"]) == set(report["groups"])

def test_rq3_equal_averages_split_repetitions_by_dataset_target(tmp_path: Path) -> None:
    """Three patient splits are fixed repetitions, not independent RQ3 cells."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import load_rq3_cells

    for split, def_val in enumerate((0.1, 0.2, 0.3)):
        write_json(
            tmp_path / f"split={split}" / "data" / "rq3.json",
            {
                "cells": [
                    {
                        "group": "tcga-ut",
                        "assignment": "native",
                        "severity": "severe",
                        "method": "ce",
                        "rho": 10.0,
                        "support_difficulty_alignment": 0.2,
                        "separability": 0.5,
                        "learnability": 0.4,
                        "log_min_support": 2.0,
                        "log_effective_support": 1.0,
                        "is_wsi": 0.0,
                        "gate_passed": True,
                        "deficit_ba": def_val,
                        "deficit_se": 0.01,
                        "recovery": np.nan,
                        "recovery_se": np.nan,
                    }
                ]
            },
        )
    write_json(
        tmp_path / "data" / "cross_split_gates_and_recovery.json",
        {
            "comparisons": [
                {
                    "assignment": "native",
                    "severity": "severe",
                    "method": "ce",
                    "gate": "discrimination",
                    "gate_passed": True,
                    "bootstrap_effect": [0.1, 0.2],
                },
                {
                    "assignment": "native",
                    "severity": "severe",
                    "method": "ce",
                    "gate": "discrimination",
                    "gate_passed": True,
                    "bootstrap_effect": [0.1, 0.2],
                },
                {
                    "assignment": "native",
                    "severity": "severe",
                    "method": "ce",
                    "gate": "discrimination",
                    "gate_passed": True,
                    "bootstrap_effect": [0.1, 0.2, 0.3],
                },
            ]
        },
    )

    cells = load_rq3_cells([tmp_path])

    assert len(cells) == 1
    assert cells[0]["group"] == "tcga-ut"
    # Replicate 0 is the observed cross-split deficit; the crossed comparison
    # that wins the key is [0.1, 0.2, 0.3], so the point estimate is 0.1, not
    # the bootstrap mean (0.2).
    assert cells[0]["deficit_ba"] == pytest.approx(0.1)

def test_rq3_cells_keep_assignment_and_severity_and_dataset_target_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """RQ3 cells retain their crossed identity and never merge a dataset's targets."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import _cells, run_rq3

    observed_regimes: list[bool] = []

    def covariates(_: dict, is_mil: bool, __: dict, *args: object) -> dict[str, float]:
        observed_regimes.append(is_mil)
        return {
            "separability": 0.5,
            "learnability": 0.4,
            "log_min_support": 2.0,
            "log_effective_support": 1.0,
            "is_wsi": 1.0,
        }

    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._covariates",
        covariates,
    )
    comparisons = [
        {
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "discrimination",
            "gate_passed": True,
            "effect": 0.1,
            "bootstrap_effect": [0.05, 0.15],
        }
    ]
    freeze = {
        "assignment_conditions": {
            "native": {
                "severe": {
                    "achieved_rho": 10.0,
                    "allocated_counts": {"A": 10, "B": 100},
                    "contribution_stats": {},
                    "path": str(tmp_path / "x.csv"),
                }
            }
        }
    }
    freeze["difficulty_evidence"] = {"difficulty": {"A": 0.1, "B": 0.2}}

    cells = _cells({"data": tmp_path}, comparisons, freeze, "panda:wsi", True)
    report = run_rq3(
        {"data": tmp_path},
        {"dataset": {"name": "panda", "regime": "patch", "target": "changed_target"}},
        {
            **freeze,
            "dataset_provenance": {
                "name": "panda",
                "regime": "wsi",
                "target": "isup_grade",
            },
        },
        comparisons,
    )

    assert cells[0]["assignment"] == "native"
    assert cells[0]["severity"] == "severe"
    assert report["cells"][0]["group"] == "panda:isup_grade"
    assert observed_regimes[-1] is True

def test_rq3_cross_split_values_come_from_crossed_bootstrap(tmp_path: Path) -> None:
    """RQ3 uses the equal-split gate and ratio distribution, not split-level averages."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import load_rq3_cells

    cell = {
        "group": "tcga-ut:patch",
        "assignment": "native",
        "severity": "severe",
        "method": "weighted_ce",
        "rho": 10.0,
        "support_difficulty_alignment": 0.2,
        "separability": 0.5,
        "learnability": 0.4,
        "log_min_support": 2.0,
        "log_effective_support": 1.0,
        "is_wsi": 0.0,
        "gate_passed": False,
        "deficit_ba": np.nan,
        "deficit_se": np.nan,
        "recovery": 0.2,
        "recovery_se": 0.01,
    }
    for split in range(3):
        write_json(tmp_path / f"split={split}" / "data" / "rq3.json", {"cells": [cell]})
    write_json(
        tmp_path / "data" / "cross_split_gates_and_recovery.json",
        {
            "comparisons": [
                {
                    "assignment": "native",
                    "severity": "severe",
                    "method": "ce",
                    "gate": "discrimination",
                    "gate_passed": True,
                    "bootstrap_effect": [0.1, 0.2],
                },
                {
                    "assignment": "native",
                    "severity": "severe",
                    "method": "weighted_ce",
                    "gate": "discrimination",
                    "gate_passed": True,
                    "bootstrap_numerator": [1.0, 4.0],
                    "bootstrap_denominator": [2.0, 2.0],
                },
            ]
        },
    )

    cells = load_rq3_cells([tmp_path])

    assert cells[0]["gate_passed"] is True
    # Observed recovery is numerator[0]/denominator[0] = 1.0/2.0 = 0.5, not the
    # mean of the per-replicate ratios (1.25). The bootstrap spread still feeds
    # the standard error.
    assert cells[0]["recovery"] == pytest.approx(0.5)
    assert cells[0]["recovery_se"] == pytest.approx(np.std([0.5, 2.0], ddof=1))

def test_gate_uses_observed_deficit_not_bootstrap_mean():
    # Replicate 0 is the observed cohort: the observed deficit is 0.01 (must not
    # open the 0.02 gate) while bootstrap replicates 1.. average ~0.03 with a CI
    # that excludes zero.
    balanced = np.array([0.51, 0.53, 0.54, 0.55])
    severity = np.array([0.50, 0.50, 0.51, 0.52])
    comparison, passed, dist = discrimination_gate_comparison(
        "moderate", balanced, severity
    )
    assert dist[0] == pytest.approx(0.01)  # observed deficit
    assert comparison["effect"] == pytest.approx(0.01)
    assert passed is False

def test_recovery_is_ratio_of_observed_points_not_mean_of_ratios():
    inp = _SeverityInputs({}, "moderate", {}, {}, None, 2, 10, 0, "native")
    # Index 0 is the observed cohort; ratio of observed points is 0.06/0.12 = 0.5,
    # whereas the mean of the two per-replicate ratios would be (0.2 + 5.0)/2 = 2.6.
    effect_dist = np.array([0.06, 0.02, 0.10])
    deficit_dist = np.array([0.12, 0.10, 0.02])
    entry = _recovery_comparison(
        inp,
        "weighted_ce",
        "discrimination",
        effect_dist,
        deficit_dist,
        gate_passed=True,
        p_value=None,
    )
    assert entry["recovery"] == pytest.approx(0.5)
    assert entry["numerator"] == pytest.approx(0.06)
    assert entry["denominator"] == pytest.approx(0.12)
