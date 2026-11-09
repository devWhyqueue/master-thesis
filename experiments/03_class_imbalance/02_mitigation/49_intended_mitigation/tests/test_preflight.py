"""Preflight gates 1, 2, 4 and the CE-reproduction summary (gate 3)."""

from __future__ import annotations

import json
import shutil

import pytest
from imbalance_benchmark.common import write_json

from analyze.paths import run_dir

from intended import lora_dirs, lr_paths
from intended.fit_lr import fit_lr_shard
from intended.preflight import run_preflight
from intended.reproduction import MISMATCH_NAME, check_ce

from intended_world import build_world, patch_freeze_loader, swap_class_order, write_damage_records, write_source_records


@pytest.fixture
def world(tmp_path, monkeypatch) -> dict:
    patch_freeze_loader(monkeypatch)
    config = build_world(tmp_path)
    write_source_records(config)
    write_damage_records(config, [(0, 0)])
    return config


def _report(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate1_passes_on_consistent_world(world: dict, tmp_path) -> None:
    report = _report(run_preflight(world, tmp_path / "report"))
    assert report["gates"]["row_identity"]["passed"], report["gates"]["row_identity"]
    assert report["gates"]["row_identity"]["checked"] == 60
    assert report["passed"]


def test_gate1_fails_on_class_order_swap(world: dict, tmp_path) -> None:
    swap_class_order(world)
    report = _report(run_preflight(world, tmp_path / "report"))
    assert not report["gates"]["row_identity"]["passed"]
    assert "class" in json.dumps(report["gates"]["row_identity"]["problems"])
    assert not report["passed"]


def test_gate2_lists_missing_record(world: dict, tmp_path) -> None:
    missing = run_dir(lora_dirs(world)[1], "r100", "gcl2", 0.05, 4)
    shutil.rmtree(missing)
    report = _report(run_preflight(world, tmp_path / "report"))
    gate = report["gates"]["records_present"]
    assert not gate["passed"] and not report["passed"]
    assert any(str(missing) in problem for problem in gate["problems"])


def test_gate4_passes(world: dict, tmp_path) -> None:
    gate = _report(run_preflight(world, tmp_path / "report"))["gates"]["la_equivalence"]
    assert gate["passed"], gate
    assert gate["ba_difference_pp"] < 0.01


def test_check_ce_reports_mismatch(world: dict, tmp_path) -> None:
    fit_lr_shard(world, 0, 0, "ce")
    other = run_dir(lr_paths(world)[2], "r1", "ce", None, 9)
    other.mkdir(parents=True)
    write_json(other / MISMATCH_NAME, {"where": "split 2 draw 9 arm r1", "differing_rows": 3})
    report = _report(check_ce(world, tmp_path / "report"))
    assert report["matched"] == 2 and not report["passed"]
    assert report["mismatches"][0]["differing_rows"] == 3
    assert len(report["missing"]) == 57
