"""Six paired contrasts: interval rules, joint verdict, test-blind selection, and input gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch
from imbalance_benchmark.common import ensure_dirs, split_paths

from analyze import select as exp40_select
from analyze.paths import run_dir as shard_run_dir
from centre import N_DRAWS, N_SPLITS
from mitigation.grid import stage1_jobs, stage2_jobs

from lora_ablation.analyze import (
    METHODS,
    analyze_dataset,
    analyze_experiment,
    joint_verdict,
    select_candidates,
    summarize_contrast,
)
from lora_ablation.fit import HEAD_NAME
from lora_ablation import source_run_dir
from lora_ablation.preflight import audit_inputs, config_digest, write_inventory
from lora_ablation.render import render_results
from lora_ablation.resume import AUDIT_SIDECAR

from world import CLASSES, build_world, make_manifest, patch_freeze_loader, write_record

SIX = [f"{d}/{m}" for d in ("bracs", "tcga_ut") for m in METHODS]


def _dist(estimate: float, replicates: np.ndarray) -> np.ndarray:
    return np.concatenate([[estimate], replicates])


def test_six_primary_methods_and_quantiles() -> None:
    assert METHODS == ("la", "posthoc_la", "cuda")
    replicates = np.random.default_rng(0).normal(0.0, 1.0, 10_000)
    result = summarize_contrast(_dist(5.0, replicates))
    low, high = np.quantile(replicates, [0.05 / 12, 1 - 0.05 / 12])
    assert (result["lower"], result["upper"]) == (low, high)


def test_observed_column_is_excluded() -> None:
    result = summarize_contrast(_dist(99.0, np.full(10_000, -0.2)))
    assert result["lower"] == result["upper"] == pytest.approx(-0.2) and result["estimate"] == 99.0


@pytest.mark.parametrize(
    "level,decision",
    [(-0.2, "noninferior"), (-1.2, "lora_advantage"), (-1.0, "inconclusive")],
)
def test_threshold_decisions(level, decision) -> None:
    assert summarize_contrast(_dist(0.0, np.full(10_000, level)))["decision"] == decision


def test_wide_crossing_interval_is_inconclusive() -> None:
    replicates = np.random.default_rng(1).normal(-1.0, 3.0, 10_000)
    assert summarize_contrast(_dist(0.0, replicates))["decision"] == "inconclusive"


@pytest.mark.parametrize("bad", ["inconclusive", "lora_advantage"])
def test_joint_requires_all_six(bad) -> None:
    ok = {key: "noninferior" for key in SIX}
    assert joint_verdict(ok) == "omission_supported"
    assert joint_verdict({**ok, "bracs/cuda": bad}) == "not_supported"
    with pytest.raises(ValueError):
        joint_verdict({k: v for k, v in ok.items() if k != "tcga_ut/la"})
    with pytest.raises(ValueError):
        joint_verdict({**ok, "bracs/crt": "noninferior"})


def _write_validation(config: dict[str, Any], paths: dict, val: dict, test: dict, stage: str, method: str) -> None:
    for s in range(N_SPLITS):
        for d in range(N_DRAWS):
            for param, ba in val.items():
                out = shard_run_dir(paths[s], "r100", method, param, d)
                write_record(out, {}, np.array([0]), np.array([0]), ba)
                record = out / "run.json"
                payload = json.loads(record.read_text())
                payload["splits"]["test"]["endpoints"] = {"patient_macro_balanced_accuracy": test[param]}
                record.write_text(json.dumps(payload))


@pytest.mark.parametrize("method,stage", [("la", "stage1"), ("posthoc_la", "stage2")])
def test_selection_is_independent_validation_only(tmp_path, monkeypatch, method, stage) -> None:
    patch_freeze_loader(monkeypatch)
    config = build_world(tmp_path, records=False)
    paths = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    params = config["mitigation"]["grid"]["r100"][stage][method]
    val = {p: 0.5 for p in params}
    val[params[-2]] = val[params[-1]] = 0.7  # tie between the last two candidates -> config order
    test = {p: float(i) for i, p in enumerate(reversed(params))}  # opposing test ranking
    _write_validation(config, paths, val, test, stage, method)
    real = exp40_select.read_run_record

    def guarded(directory, splits=None, array_fields=None):
        assert splits == ("validation",), "selection opened a non-validation split"
        return real(directory, splits=splits, array_fields=array_fields)

    monkeypatch.setattr(exp40_select, "read_run_record", guarded)
    assert set(select_candidates(config, paths, "r100", method).values()) == {params[-2]}


def _frozen(config: dict[str, Any], fingerprint: str, preds: Callable[[np.ndarray], np.ndarray]) -> None:
    """Complete frozen outputs for every candidate: records, sidecars, and one CE head per draw."""
    manifest = make_manifest(Path("."), Path("."))
    test_y = np.array([CLASSES.index(c) for c in manifest.query("split == 'test'")["cancer_type"]])
    for s in range(N_SPLITS):
        paths = split_paths(ensure_dirs(config), s)
        for d in range(N_DRAWS):
            ce = shard_run_dir(paths, "r100", "ce", None, d)
            ce.mkdir(parents=True, exist_ok=True)
            torch.save({"weight": torch.zeros(2, 4), "bias": torch.zeros(2)}, ce / HEAD_NAME)
            head_sha = hashlib.sha256((ce / HEAD_NAME).read_bytes()).hexdigest()
            for arm, method, param in (*stage1_jobs(config), *stage2_jobs(config)):
                out = shard_run_dir(paths, arm, method, param, d)
                write_record(out, {}, test_y, preds(test_y))
                meta = {"audit_fingerprint": fingerprint, "config_sha256": config_digest(config), "init_seed": 0,
                        "resources": {"seconds": 2.0, "peak_rss_mb": 100.0}}
                if method == "posthoc_la":
                    meta["ce_head_sha256"] = head_sha
                (out / AUDIT_SIDECAR).write_text(json.dumps(meta))


@pytest.fixture
def audited(tmp_path, monkeypatch):
    patch_freeze_loader(monkeypatch)
    config = build_world(tmp_path)
    inventory = audit_inputs(config)
    write_inventory(config, inventory)
    return config, inventory["fingerprint"]


def test_identical_predictions_give_zero_contrasts(audited) -> None:
    config, fingerprint = audited
    _frozen(config, fingerprint, lambda y: y)
    block = analyze_dataset(config)
    assert set(block["primary"]) == set(METHODS)
    for result in block["primary"].values():
        assert result["estimate"] == result["lower"] == result["upper"] == 0.0
        assert result["decision"] == "noninferior"
    assert block["replicates"] == 10_000
    assert block["selection_frequency"]["frozen"]["la"] == {"0.5": 30}
    assert block["ce_relative_gain"]["frozen"]["cuda"]["estimate"] == 0.0
    assert block["resources"]["cuda"]["fits"] == 90 and block["resources"]["posthoc_la"]["fits"] == 120
    assert '"r1"' not in json.dumps(block)  # no balanced arm anywhere


def test_clear_lora_advantage_is_reported(audited) -> None:
    config, fingerprint = audited
    _frozen(config, fingerprint, lambda y: 1 - y)
    block = analyze_dataset(config)
    assert {r["decision"] for r in block["primary"].values()} == {"lora_advantage"}
    assert block["primary"]["la"]["estimate"] == pytest.approx(-100.0)


def _break(config: dict[str, Any], what: str) -> None:
    paths = split_paths(ensure_dirs(config), 2)
    if what == "record":
        (shard_run_dir(paths, "r100", "cuda", 0.5, 9) / "run.json").unlink()
    elif what == "sidecar":
        out = shard_run_dir(paths, "r100", "la", 1.0, 3) / AUDIT_SIDECAR
        out.write_text(json.dumps({**json.loads(out.read_text()), "audit_fingerprint": "other"}))
    elif what == "ce_weights":
        (shard_run_dir(paths, "r100", "ce", None, 5) / HEAD_NAME).unlink()
    elif what == "ce_changed":
        torch.save({"weight": torch.ones(2, 4), "bias": torch.zeros(2)}, shard_run_dir(paths, "r100", "ce", None, 5) / HEAD_NAME)
    else:
        record = source_run_dir(config, 1, "r100", "posthoc_la", 1.5, 4) / "run.json"
        record.write_text(record.read_text() + " ")


@pytest.mark.parametrize(
    "what,message",
    [("record", "missing"), ("sidecar", "inventory"), ("ce_weights", "CE"), ("ce_changed", "CE"), ("adapted", "inventory")],
)
def test_incomplete_or_mismatched_inputs_reject_analysis(audited, what, message) -> None:
    config, fingerprint = audited
    _frozen(config, fingerprint, lambda y: y)
    _break(config, what)
    with pytest.raises(ValueError, match=message):
        analyze_dataset(config)


def test_experiment_decides_exactly_six_contrasts(tmp_path, monkeypatch) -> None:
    patch_freeze_loader(monkeypatch)
    configs = []
    for dataset in ("bracs", "tcga_ut"):
        root = tmp_path / dataset
        root.mkdir()
        config = build_world(root, dataset)
        inventory = audit_inputs(config)
        write_inventory(config, inventory)
        _frozen(config, inventory["fingerprint"], lambda y: y)
        configs.append(config)
    analysis = analyze_experiment(configs)
    assert sorted(analysis["decisions"]) == sorted(SIX)
    assert analysis["joint_verdict"] == "omission_supported"
    assert analysis["quantiles"] == [0.05 / 12, 1 - 0.05 / 12]
    text = render_results(analysis)
    assert text.count("noninferior") >= 6 and "99.17" in text
    for banned in ("cRT", "GCL", "98.75", "balanced gain", "recovery share"):
        assert banned not in text
    with pytest.raises(ValueError, match="both datasets"):
        analyze_experiment(configs[:1])
