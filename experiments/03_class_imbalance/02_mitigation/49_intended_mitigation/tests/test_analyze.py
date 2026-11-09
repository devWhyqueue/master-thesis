"""Contrasts, decision rule, input refusals, and rendering of the exp-49 analysis."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from imbalance_benchmark.common import write_json

from analyze.paths import run_dir, shard_keys

from intended import lr_paths, sgd_paths
from intended.analyze import analyze_dataset, contrasts, share_summary, summarize
from intended.render import render_results

from intended_world import build_world, patch_freeze_loader, manifest_test_labels, write_record, write_source_records

R = 2_000


def _dist(point: float, replicates: np.ndarray) -> np.ndarray:
    return np.concatenate([[point], replicates])


@pytest.mark.parametrize(
    ("point", "level", "verdict"),
    [(1.2, 0.1, "material"), (0.9, 0.5, "not_material"), (1.2, -0.1, "not_material")],
)
def test_summarize_rule(point: float, level: float, verdict: str) -> None:
    replicates = np.linspace(level, level + 2.0, R)  # 2.5 % quantile sits just above ``level``
    result = summarize(_dist(point, replicates))
    assert result["verdict"] == verdict and result["point"] == point
    assert result["lower"] == pytest.approx(np.quantile(replicates, 0.025))


def test_share_and_full_recovery() -> None:
    assert share_summary(_dist(0.9, np.linspace(0.6, 1.2, R)))["full_recovery"]
    assert not share_summary(_dist(0.5, np.linspace(0.3, 0.7, R)))["full_recovery"]


def _dists(rng: np.random.Generator) -> dict[str, np.ndarray]:
    labels = ["ce:r1", "ce:r100", "la:r100", "bs:r100", "lora_ce:r100", "sgd_ce:r1", "sgd_ce:r100",
              "sgd_cuda:r1", "sgd_cuda:r100", "cpu_sgd_ce:r100"]
    labels += [f"{m}:{a}" for m in ("disalign", "crt", "gcl", "gcl2") for a in ("r1", "r100")]
    return {label: rng.normal(50.0, 1.0, R + 1) for label in labels}


def test_g_zero_for_la_bs() -> None:
    result = contrasts(_dists(np.random.default_rng(0)))["methods"]
    for method in ("la", "bs"):
        assert result[method]["G"]["point"] == 0.0 and result[method]["G_by_construction"]
        assert result[method]["R_minus_G"] == result[method]["R"]
    assert not result["disalign"]["G_by_construction"]


def test_cuda_uses_sgd_baseline() -> None:
    dists = _dists(np.random.default_rng(1))
    before = contrasts(dists)["methods"]["cuda"]
    dists["ce:r100"] = dists["ce:r100"] + 5.0
    dists["ce:r1"] = dists["ce:r1"] - 3.0
    after = contrasts(dists)
    assert after["methods"]["cuda"] == before
    assert after["methods"]["la"]["R"]["point"] != contrasts(_dists(np.random.default_rng(1)))["methods"]["la"]["R"]["point"]
    expected = dists["sgd_cuda:r100"][0] - dists["sgd_ce:r100"][0]
    assert after["methods"]["cuda"]["R"]["point"] == pytest.approx(expected)


@pytest.fixture
def world(tmp_path, monkeypatch) -> dict:
    patch_freeze_loader(monkeypatch)
    config = build_world(tmp_path)
    write_source_records(config)
    labels = manifest_test_labels(config)
    for roots, stage in ((lr_paths(config), "lr"), (sgd_paths(config), "sgd")):
        for arm, stages in config["mitigation"]["grid"].items():
            for method, params in stages[stage].items():
                for s, d in shard_keys():
                    for param in params:
                        write_record(run_dir(roots[s], arm, method, param, d), labels, labels)
    report = tmp_path / "report"
    write_json(report / "preflight_bracs.json", {"passed": True})
    write_json(report / "ce_reproduction_bracs.json", {"passed": True})
    return {"config": config, "report": report}


def test_analysis_runs_and_renders(world: dict) -> None:
    analysis = analyze_dataset(world["config"], world["report"])
    assert analysis["methods"]["crt"]["R"]["point"] == pytest.approx(0.0)
    methods = ("la", "bs", "disalign", "crt", "gcl", "gcl2", "sgd_cuda")
    assert set(analysis["selection_frequency"]) >= {f"{m}:r100" for m in methods} | {"sgd_cuda:r1"}
    assert sum(analysis["selection_frequency"]["la:r100"].values()) == 30
    tex = render_results({"datasets": {"bracs": analysis, "tcga_ut": analysis}})
    assert "BRACS" in tex and "TCGA-UT" in tex and "GCL2" in tex


def test_label_mismatch_refused(world: dict) -> None:
    labels = manifest_test_labels(world["config"])
    wrong = labels[::-1].copy()
    write_record(run_dir(sgd_paths(world["config"])[2], "r100", "ce", None, 5), wrong, wrong)
    with pytest.raises(ValueError, match=r"shard \(2, 5\)"):
        analyze_dataset(world["config"], world["report"])


def test_missing_record_refused(world: dict) -> None:
    missing = run_dir(lr_paths(world["config"])[1], "r100", "la", 1.5, 3)
    shutil.rmtree(missing)
    with pytest.raises(ValueError, match="missing") as error:
        analyze_dataset(world["config"], world["report"])
    assert str(missing) in str(error.value)


def test_failed_gate_refused(world: dict) -> None:
    write_json(world["report"] / "ce_reproduction_bracs.json", {"passed": False})
    with pytest.raises(ValueError, match="ce_reproduction"):
        analyze_dataset(world["config"], world["report"])
