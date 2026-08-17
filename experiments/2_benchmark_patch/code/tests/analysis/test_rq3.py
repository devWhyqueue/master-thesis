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
from imbalance_benchmark.analysis.predictors import (
    rq3_analysis,
    rq3_features,
    rq3_wiring,
)
from imbalance_benchmark.analysis.predictors.hierarchical_models import _log_scale_prior
from imbalance_benchmark.analysis.predictors.rq3_analysis import (
    _cells,
    _covariates,
)
from imbalance_benchmark.analysis.predictors.rq3_cross_split import _comparison_maps
from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    fit_deficit_model,
    fit_recovery_model,
)


def test_patch_feature_frame_gathers_resident_rows_without_sample_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RQ3 must gather resident rows once, not materialize every patch item."""

    class Patches:
        rows = object()

        def __len__(self) -> int:  # pragma: no cover - must not run
            raise AssertionError("feature_frame must not index every patch")

        def get_int_targets(self) -> np.ndarray:
            return np.array([1, 0])

    monkeypatch.setattr(
        rq3_features, "load_training_dataset", lambda *_, **__: Patches()
    )
    monkeypatch.setattr(
        rq3_features,
        "bank_index",
        lambda _: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    features, targets = rq3_features.feature_frame(
        tmp_path / "manifest.csv", None, False, None
    )

    np.testing.assert_array_equal(features, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(targets, [1, 0])


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
        "independent_shortage": 0.3,
        "support_difficulty_alignment": 0.2,
        "diversity_shortage": 0.1,
        "gate_passed": gate,
        "deficit_ba": deficit,
        "deficit_se": 0.01,
        "recovery": 0.5,
        "recovery_se": 0.1,
    }


def test_rq3_shortages_compare_only_classes_deprived_by_allocation() -> None:
    balanced = {
        "allocated_counts": {"A": 10, "B": 10},
        "contribution_stats": {
            "A": {"n_patients": 8},
            "B": {"n_patients": 8},
        },
    }
    imbalanced = {
        "allocated_counts": {"A": 5, "B": 15},
        "contribution_stats": {
            "A": {"n_patients": 4},
            "B": {"n_patients": 10},
        },
    }

    independent = rq3_features._independent_shortage(balanced, imbalanced, False)
    diversity = rq3_features._diversity_shortage(
        balanced,
        imbalanced,
        {0: 4.0, 1: 4.0},
        {0: 2.0, 1: 8.0},
        ["A", "B"],
    )

    assert independent == pytest.approx(np.log(2.0))
    assert diversity == pytest.approx(np.log(2.0))


def test_rq3_predictor_matrix_contains_exactly_four_signal_columns() -> None:
    cells = [_rq3_cell("dataset:target", "ce", 10.0, 0.1, True)]

    predictors, _ = rq3_wiring.build_predictors(cells)

    assert predictors.shape == (1, 4)
    np.testing.assert_allclose(predictors[0], [np.log(10.0), 0.3, 0.2, 0.1])


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


def test_rq3_damage_and_recovery_models_run_with_four_predictors():
    rng = np.random.default_rng(0)
    groups = [f"dataset_{i % 4}" for i in range(24)]
    cells = []
    for i in range(24):
        rho = float(rng.choice([1.0, 10.0, 100.0]))
        diversity_shortage = float(rng.normal())
        gate_passed = rho > 1.0
        cells.append(
            {
                "group": groups[i],
                "rho": rho,
                "independent_shortage": float(rng.normal()),
                "support_difficulty_alignment": 0.2,
                "diversity_shortage": diversity_shortage,
                "gate_passed": gate_passed,
                "deficit_ba": float(rng.normal(0.05, 0.02)),
                "deficit_se": 0.01,
                "recovery": float(rng.normal(0.5, 0.1)),
                "recovery_se": 0.05,
            }
        )
    deficit_model = fit_deficit_model(cells)
    recovery_model = fit_recovery_model(cells)
    assert len(deficit_model["slopes"]) == 4
    assert len(recovery_model["slopes"]) == 4


def test_rq3_recovery_model_empty_when_no_gated_cells():
    cells = [
        {
            "group": "g",
            "rho": 1.0,
            "gate_passed": False,
            "recovery": 0.0,
            "recovery_se": 0.01,
        }
    ]
    assert fit_recovery_model(cells) == {}


def test_rq3_cells_keep_calibration_gate_recovery(monkeypatch: pytest.MonkeyPatch):
    """A calibration-only cell must use tail-NLL recovery, not BA recovery."""
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._reference_block",
        lambda *_: {},
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._covariates",
        lambda *_: {"independent_shortage": 0.3, "diversity_shortage": 0.1},
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
        "conditions": {"balanced": {}},
        "construction_seed": 7,
        "difficulty_evidence": {"difficulty": {"A": 0.1, "B": 0.2}},
        "assignment_conditions": {
            "native": {
                "severe": {
                    "achieved_rho": 100.0,
                    "allocated_counts": {"A": 10, "B": 100},
                }
            }
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


def test_rq3_reference_uses_validation_only_to_seed_feature_bank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded: list[tuple[str, str | None]] = []

    def feature_frame(
        path: Path, split: str | None, *_: object
    ) -> tuple[np.ndarray, np.ndarray]:
        loaded.append((path.name, split))
        return np.array([[0.0], [1.0]]), np.array([0, 1])

    monkeypatch.setattr(rq3_features, "feature_frame", feature_frame)
    monkeypatch.setattr(
        rq3_features,
        "_fixed_diversity",
        lambda *_: {0: 1.0, 1: 1.0},
    )
    balanced = {"path": str(tmp_path / "manifest_balanced.csv")}

    reference = rq3_features._reference_block(
        {"data": tmp_path}, False, ["A", "B"], balanced, 7
    )

    assert loaded == [("manifest.csv", "validation")]
    assert reference["condition"] is balanced
    assert reference["diversity"] == {0: 1.0, 1: 1.0}


def test_rq3_covariates_contain_only_two_shortage_contrasts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    condition = tmp_path / "manifest_native_severe.csv"
    condition.touch()
    balanced_meta = {
        "allocated_counts": {"A": 10, "B": 10},
        "contribution_stats": {
            "A": {"n_patients": 8},
            "B": {"n_patients": 8},
        },
    }
    condition_meta = {
        "path": str(condition),
        "allocated_counts": {"A": 5, "B": 15},
        "contribution_stats": {
            "A": {"n_patients": 4},
            "B": {"n_patients": 10},
        },
    }
    monkeypatch.setattr(rq3_features, "_fixed_diversity", lambda *_: {0: 2.0, 1: 8.0})
    result = rq3_analysis._covariates(
        {"data": tmp_path},
        False,
        condition_meta,
        {"condition": balanced_meta, "diversity": {0: 4.0, 1: 4.0}, "seed": 7},
        {"class_names": ["A", "B"]},
    )

    assert set(result) == {"independent_shortage", "diversity_shortage"}
    assert result["independent_shortage"] == pytest.approx(np.log(2.0))
    assert result["diversity_shortage"] == pytest.approx(np.log(2.0))


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
    """Combined RQ3 contains only damage, recovery, and damage stability fits."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import cross_dataset_rq3

    cells = []
    for i, group in enumerate(["tcga:patch", "tcga:wsi", "bracs:patch", "panda:wsi"]):
        cells.append(_rq3_cell(group, "ce", 10.0 + i, 0.05 + 0.01 * i, gate=True))
        cells.append(_rq3_cell(group, "weighted_ce", 10.0 + i, np.nan, gate=True))

    report = cross_dataset_rq3(cells)

    assert report["n_groups"] == 4
    assert set(report["models"]) == {"damage", "recovery"}
    assert len(report["models"]["damage"]["rand_intercepts"]) == 4
    assert "sensitivity" not in report
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
                        "independent_shortage": 0.3,
                        "support_difficulty_alignment": 0.2,
                        "diversity_shortage": 0.1,
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
    assert cells[0]["independent_shortage"] == pytest.approx(0.3)
    assert cells[0]["diversity_shortage"] == pytest.approx(0.1)
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
            "independent_shortage": 0.3,
            "diversity_shortage": 0.1,
        }

    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._reference_block",
        lambda *_: {},
    )
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
        "conditions": {"balanced": {}},
        "construction_seed": 7,
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
        "independent_shortage": 0.3,
        "support_difficulty_alignment": 0.2,
        "diversity_shortage": 0.1,
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
