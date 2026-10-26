"""Preflight: v2 workload counts, protocol/stage contract, source coverage, identity, prerequisites."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from imbalance_benchmark.common import write_run_record

from lora_ablation import (
    STAGE1_RECIPES,
    load_protocol_config,
    protocol_grid,
    workload_counts,
)
from lora_ablation import source_run_dir
from lora_ablation.preflight import audit_inputs

from world import CHECKPOINT, PINNED, build_world, patch_freeze_loader

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def world(tmp_path, monkeypatch):
    patch_freeze_loader(monkeypatch)
    return build_world(tmp_path)


def test_workload_counts() -> None:
    bracs = workload_counts(load_protocol_config(_CONFIGS / "bracs.yaml"))
    tcga = workload_counts(load_protocol_config(_CONFIGS / "tcga_ut.yaml"))
    assert tuple(bracs.values()) == (90, 90, 120, 300, 300)
    assert tuple(tcga.values()) == (120, 90, 150, 360, 360)
    assert list(bracs) == ["cpu_fits", "cuda_fits", "score_candidates", "records", "source_records"]


def test_protocol_grids_and_stage1_recipe() -> None:
    for name, steps in (("bracs", 600), ("tcga_ut", 2400)):
        config = load_protocol_config(_CONFIGS / f"{name}.yaml")
        assert list(config["mitigation"]["grid"]) == ["r100"]
        assert config["mitigation"]["grid"]["r100"] == protocol_grid(name)
        assert STAGE1_RECIPES[name]["steps"] == config["mitigation"]["train"]["stage1"]["steps"] == steps
        assert config["protocol"]["id"] == "strong-mitigations-v2"
    assert protocol_grid("bracs")["stage1"]["la"] == [0.5, 1.0]
    assert protocol_grid("tcga_ut")["stage2"]["posthoc_la"] == [0.25, 0.5, 1.0, 1.5, 2.0]


def test_inventory_on_valid_world(world) -> None:
    inventory = audit_inputs(world)
    assert inventory["workload"]["records"] == 300
    assert len(inventory["source_records"]) == 300
    assert inventory["checkpoint"] == CHECKPOINT
    json.dumps(inventory)
    assert audit_inputs(world)["fingerprint"] == inventory["fingerprint"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["protocol"].update(id="cRT-GCL-v1"),
        lambda c: c["paths"].update(outputs=c["paths"]["outputs"].replace("strong_mitigations_v2", "old")),
        lambda c: c["mitigation"]["grid"].update(r1=c["mitigation"]["grid"]["r100"]),
        lambda c: c["mitigation"]["grid"]["r100"]["stage2"].update(crt=[None]),
        lambda c: c["mitigation"]["grid"]["r100"]["stage1"]["la"].append(2.0),
        lambda c: c["source_grid"]["r100"]["stage1"]["cuda"].pop(),
        lambda c: c["mitigation"]["train"]["stage1"].update(steps=2400),
        lambda c: c["mitigation"]["cuda"].update(gamma=0.5),
    ],
)
def test_legacy_or_drifted_protocol_fails(world, mutate) -> None:
    drifted = copy.deepcopy(world)
    mutate(drifted)
    with pytest.raises(ValueError):
        audit_inputs(drifted)


@pytest.mark.parametrize(
    "stage,method,param",
    [("stage1", "ce", None), ("stage1", "cuda", 0.5), ("stage2", "posthoc_la", 1.5)],
)
def test_missing_source_candidate_fails(world, stage, method, param) -> None:
    assert stage in world["mitigation"]["grid"]["r100"]
    (source_run_dir(world, 1, "r100", method, param, 3) / "run.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        audit_inputs(world)


def test_changed_source_record_changes_fingerprint(world) -> None:
    before = audit_inputs(world)["fingerprint"]
    record = source_run_dir(world, 0, "r100", "ce", None, 0) / "run.json"
    record.write_text(record.read_text() + " ")
    assert audit_inputs(world)["fingerprint"] != before


def test_pairing_checks_identity_not_only_labels(world) -> None:
    path = Path(world["slurm"]["exp2_outputs"]) / "split=0" / "data" / "manifest.csv"
    df = pd.read_csv(path)
    test = df["split"] == "test"
    df.loc[test, "case_id"] = df.loc[test, "case_id"].replace({"Ate0": "Ate1", "Ate1": "Ate0"})
    df.to_csv(path, index=False)  # labels unchanged, identities permuted
    with pytest.raises(ValueError, match="identity"):
        audit_inputs(world)


def test_labels_must_match_manifest(world) -> None:
    write_run_record(
        source_run_dir(world, 0, "r100", "la", 0.5, 0),
        {"class_counts": {}, "splits": {"test": {"labels": [0], "preds": [0]}}},
    )
    with pytest.raises(ValueError, match="labels"):
        audit_inputs(world)


@pytest.mark.parametrize(
    "field,value",
    [("encoder_revision", "deadbeef"), ("weights_sha256", "00"), ("feature_dim", 1536), ("dtype", "float32")],
)
def test_feature_provenance_fails(world, tmp_path, field, value) -> None:
    (tmp_path / "features" / "feature_provenance.json").write_text(json.dumps({**PINNED, field: value}))
    with pytest.raises(ValueError, match="provenance"):
        audit_inputs(world)


def test_missing_feature_provenance_fails(world, tmp_path) -> None:
    (tmp_path / "features" / "feature_provenance.json").unlink()
    with pytest.raises(ValueError, match="provenance"):
        audit_inputs(world)


def test_missing_freeze_identity_fails(world) -> None:
    freeze = Path(world["slurm"]["exp2_outputs"]) / "split=2" / "data" / "manifest_freeze.json"
    freeze.write_text(json.dumps({"class_names": ["A", "B"]}))
    with pytest.raises(ValueError, match="identity"):
        audit_inputs(world)


def test_missing_raw_image_fails(world, tmp_path) -> None:
    (tmp_path / "images" / "Atr0_0.jpg").unlink()
    with pytest.raises(ValueError, match="raw image"):
        audit_inputs(world)


def test_checkpoint_evidence_required(world, monkeypatch) -> None:
    from lora_ablation import preflight

    monkeypatch.setattr(preflight, "cached_checkpoint", lambda: {**CHECKPOINT, "weights_sha256": "00"})
    with pytest.raises(ValueError, match="checkpoint"):
        audit_inputs(world)

    def uncached() -> dict:
        raise ValueError("pinned Virchow2 checkpoint is not cached")

    monkeypatch.setattr(preflight, "cached_checkpoint", uncached)
    with pytest.raises(ValueError, match="checkpoint"):
        audit_inputs(world)
