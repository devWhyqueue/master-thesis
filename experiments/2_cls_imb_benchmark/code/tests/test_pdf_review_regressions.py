"""Regressions for divergences found reviewing 2_cls_imb_benchmark.pdf.

Each test fails against the pre-fix implementation and passes once the matching
correction is in place.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from imbalance_benchmark.common import sign_file


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


def _write_freeze_fixture(tmp_path: Path) -> Path:
    """Config + three signed patient-split manifests, ready for cmd_freeze."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"outputs": str(tmp_path / "outputs")},
                "dataset": {
                    "name": "synthetic",
                    "regime": "patch",
                    "target": "synthetic_target",
                    "version": "test-fixture-v1",
                    "eligibility_rules": {"fixture": True},
                },
                "analysis": {"bootstrap_replicates": 2},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "case_id": f"{cls}_{index}",
            "slide_id": f"{cls}_{index}",
            "patch_id": f"{cls}_{index}_patch",
            "cancer_type": cls,
            "split": "train" if index < 30 else "test",
        }
        for cls in ("A", "B")
        for index in range(40)
    ]
    for split_index in range(3):
        data_dir = tmp_path / "outputs" / f"split={split_index}" / "data"
        data_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(data_dir / "manifest.csv", index=False)
        pilot = data_dir / "pilot_report.json"
        pilot.write_text(
            json.dumps({"definitive_floor": 10, "quotas": {"0": 1}, "excluded": False}),
            encoding="utf-8",
        )
        sign_file(pilot)
    return config_path


def test_freeze_rejects_pilot_edited_after_signing(tmp_path: Path) -> None:
    """A pilot edited after signing must not be baked into the definitive freeze.

    Finding: "Pilot evidence is not locked into the definitive freeze." The floor
    is read from the pilot at freeze time, so freezing must verify the pilot's
    signature first; otherwise a post-signing edit silently sets the frozen floor.
    """
    from imbalance_benchmark.commands.freeze import cmd_freeze

    config_path = _write_freeze_fixture(tmp_path)
    tampered = tmp_path / "outputs" / "split=0" / "data" / "pilot_report.json"
    tampered.write_text(  # altered content, stale signature
        json.dumps({"definitive_floor": 1, "quotas": {"0": 1}, "excluded": False}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="signed lock"):
        cmd_freeze(Namespace(config=str(config_path), seed=7, split_index=0))


def test_exploratory_methods_are_not_hypothesis_tested() -> None:
    """Exploratory methods keep effects/CIs but carry no p-value or "tested" status.

    Finding: "Exploratory methods receive hypothesis tests." Setup §3.6 limits
    hypothesis tests to the four primary methods.
    """
    from imbalance_benchmark.analysis.inference.holm import apply_holm

    out = apply_holm(
        [
            {
                "method": "cfal",
                "gate": "discrimination",
                "severity": "severe",
                "gate_passed": True,
                "p_value": 0.02,
            }
        ]
    )
    row = out[0]
    assert row["family"] == "exploratory"
    assert row["status"] != "tested"
    assert row["p_value"] is None
    assert row["adjusted_p_value"] is None


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
    weight decay, dropout, and checkpoint interval were source-only.
    """
    from imbalance_benchmark.modeling.training.config import resolve_training_config

    patch = resolve_training_config({}, is_mil=False)
    assert patch["optimizer"] == "AdamW"
    assert patch["weight_decay"] == 1e-4
    assert patch["batch_size"] == 128
    assert patch["checkpoint_interval"] == 50
    assert patch["dropout"] == 0.1
    assert resolve_training_config({}, is_mil=True)["batch_size"] == 32


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
