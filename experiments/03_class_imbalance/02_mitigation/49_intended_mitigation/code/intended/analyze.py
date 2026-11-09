"""Recovery R, share R/D, balanced gain G and R - G of every method against the frozen CE.

Every distribution comes from exp-40's validation-only ``select`` and patient-macro BA bootstrap
(``selected_ba``) pooled with exp-45's shared design (contexts, canonical class order, one set
of draw weights), so all contrasts are paired. Column zero is the observed estimate; intervals
use the replicate columns. Frozen-feature and LoRA methods are measured against the frozen
logistic-regression CE; CUDA against CE_SGD through its own code path. Analysis refuses inputs
whose gates failed, any missing record, and any shard whose records disagree on test labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import read_run_record, write_json

from analyze.bootstrap import selected_ba, selected_thirds
from analyze.paths import run_dir, shard_keys
from analyze.select import candidate_params, select, selected_frequency

from centre.analyze import pooled

from lora_ablation.analyze import _Design

from intended import (
    ARMS,
    DATASETS,
    REPORT_DIR,
    cuda_r100_dirs,
    lora_dirs,
    lr_paths,
    sgd_paths,
)
from intended.render import render_results

__all__ = ["analyze_dataset", "contrasts", "run_analyze", "share_summary", "summarize"]

MATERIAL_PP = 1.0
QUANTILES = (0.025, 0.975)
FROZEN_METHODS = ("la", "bs", "disalign")
LORA_METHODS = ("crt", "gcl", "gcl2")
G_BY_CONSTRUCTION = ("la", "bs")

Spec = tuple[dict[str, Any], dict[int, dict[str, Path]], str, str, str]


def _interval(distribution: np.ndarray) -> dict[str, float]:
    distribution = np.asarray(distribution, dtype=np.float64)
    lower, upper = (float(q) for q in np.quantile(distribution[1:], QUANTILES))
    return {"point": float(distribution[0]), "lower": lower, "upper": upper}


def summarize(distribution: np.ndarray) -> dict[str, float | str]:
    """Point, 95 % interval, and the material rule: point >= 1 pp and lower bound > 0."""
    result = _interval(distribution)
    material = result["point"] >= MATERIAL_PP and result["lower"] > 0
    return {**result, "verdict": "material" if material else "not_material"}


def share_summary(distribution: np.ndarray) -> dict[str, Any]:
    """Share R/D with its interval; full recovery iff the interval contains 1."""
    result = _interval(distribution)
    return {**result, "full_recovery": result["lower"] <= 1.0 <= result["upper"]}


def _block(r: np.ndarray, d: np.ndarray, g: np.ndarray) -> dict[str, Any]:
    return {
        "R": summarize(r),
        "share": share_summary(r / d),
        "G": summarize(g),
        "R_minus_G": summarize(r - g),
    }


def contrasts(dists: dict[str, np.ndarray]) -> dict[str, Any]:
    """Every contrast of one dataset from its pooled BA distributions (keys ``method:arm``)."""
    d_lr = dists["ce:r1"] - dists["ce:r100"]
    methods: dict[str, Any] = {}
    for m in FROZEN_METHODS + LORA_METHODS:
        r = dists[f"{m}:r100"] - dists["ce:r100"]
        by_construction = m in G_BY_CONSTRUCTION
        g = np.zeros_like(r) if by_construction else dists[f"{m}:r1"] - dists["ce:r1"]
        methods[m] = {**_block(r, d_lr, g), "G_by_construction": by_construction}
    d_sgd = dists["sgd_ce:r1"] - dists["sgd_ce:r100"]
    cuda_r = dists["sgd_cuda:r100"] - dists["sgd_ce:r100"]
    methods["cuda"] = _block(cuda_r, d_sgd, dists["sgd_cuda:r1"] - dists["sgd_ce:r1"])
    ba = {
        k: _interval(v) for k, v in dists.items() if k.split(":")[0] in ("ce", "sgd_ce")
    }
    return {
        "baseline": {"ba": ba, "D": summarize(d_lr), "D_sgd": summarize(d_sgd)},
        "methods": methods,
        "sensitivity": _sensitivity(dists),
    }


def _sensitivity(dists: dict[str, np.ndarray]) -> dict[str, Any]:
    """R against exp-41/43 LoRA CE, and exp-45's CPU CE_SGD r100 minus the new GPU CE_SGD r100."""
    lora_ce = {
        m: summarize(dists[f"{m}:r100"] - dists["lora_ce:r100"])
        for m in FROZEN_METHODS + LORA_METHODS
    }
    cpu_gap = dists["cpu_sgd_ce:r100"] - dists["sgd_ce:r100"]
    return {"R_vs_lora_ce": lora_ce, "cpu_minus_gpu_ce_sgd_r100": summarize(cpu_gap)}


def _specs(config: dict[str, Any]) -> dict[str, Spec]:
    """Label -> (grid config, per-split roots, arm, stage, method) of every distribution."""
    roots = {"lr": lr_paths(config), "sgd": sgd_paths(config)}
    specs: dict[str, Spec] = {
        f"{'sgd_' if stage == 'sgd' else ''}{m}:{arm}": (
            config,
            roots[stage],
            arm,
            stage,
            m,
        )
        for arm, stages in config["mitigation"]["grid"].items()
        for stage in ("lr", "sgd")
        for m in stages[stage]
    }
    lora = {"mitigation": {"grid": config["sources"]["mitigation_grid"]}}
    for arm in ARMS:
        for m, stage in (("crt", "stage2"), ("gcl", "stage1"), ("gcl2", "stage2")):
            specs[f"{m}:{arm}"] = (lora, lora_dirs(config), arm, stage, m)
    specs["lora_ce:r100"] = (lora, lora_dirs(config), "r100", "stage1", "ce")
    frozen = {"mitigation": {"grid": {"r100": config["sources"]["frozen_cuda_grid"]}}}
    specs["sgd_cuda:r100"] = (frozen, cuda_r100_dirs(config), "r100", "stage1", "cuda")
    specs["cpu_sgd_ce:r100"] = (frozen, cuda_r100_dirs(config), "r100", "stage1", "ce")
    return specs


def _require_gates(config: dict[str, Any], report_dir: Path) -> None:
    for name in ("preflight", "ce_reproduction"):
        path = report_dir / f"{name}_{config['dataset']['name']}.json"
        if not path.is_file() or not json.loads(path.read_text(encoding="utf-8")).get(
            "passed"
        ):
            raise ValueError(f"{name} did not pass (or is missing): {path}")


def _has_labels(directory: Path) -> bool:
    record = read_run_record(directory, splits=("test",), array_fields=("labels",))
    return record is not None and "labels" in record["splits"].get("test", {})


def _require_records(specs: dict[str, Spec]) -> None:
    """Every candidate record must exist with its test labels; else list the directories."""
    missing = [
        str(run_dir(roots[s], arm, method, param, d))
        for cfg, roots, arm, stage, method in specs.values()
        for s, d in shard_keys()
        for param in candidate_params(cfg, arm, stage, method)
        if not _has_labels(run_dir(roots[s], arm, method, param, d))
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} records missing or without test labels:\n"
            + "\n".join(missing)
        )


def _labels(directory: Path) -> np.ndarray:
    record = (
        read_run_record(directory, splits=("test",), array_fields=("labels",)) or {}
    )
    return np.asarray(record["splits"]["test"]["labels"])


def _require_paired(dirs: dict[str, dict[tuple[int, int], Path]]) -> None:
    for shard in shard_keys():
        reference = _labels(dirs["ce:r100"][shard])
        for label, by_shard in dirs.items():
            if not np.array_equal(_labels(by_shard[shard]), reference):
                raise ValueError(
                    f"shard {shard}: {label} test labels differ from the CE record"
                )


def _selected_dirs(specs: dict[str, Spec]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validation-only selection per label and shard, and each winner's run directory."""
    _require_records(specs)
    selections = {label: select(*spec) for label, spec in specs.items()}
    dirs = {
        label: {
            (s, d): run_dir(specs[label][1][s], specs[label][2], specs[label][4], p, d)
            for (s, d), p in sel.items()
        }
        for label, sel in selections.items()
    }
    return selections, dirs


def analyze_dataset(
    config: dict[str, Any], report_dir: Path = REPORT_DIR
) -> dict[str, Any]:
    """All contrasts, selection frequencies and rank-third recall of one dataset."""
    _require_gates(config, report_dir)
    selections, dirs = _selected_dirs(_specs(config))
    _require_paired(dirs)
    design = _Design(config)
    k = len(design.names)
    dists = {
        label: pooled(selected_ba(by, design.ctxs, design.perms, k), design.weights)
        for label, by in dirs.items()
    }
    r100 = [label for label in dirs if label.endswith(":r100")]
    return {
        "dataset": config["dataset"]["name"],
        "replicates": design.ctxs[0].n_replicates - 1,
        **contrasts(dists),
        "selection_frequency": {
            label: selected_frequency(sel) for label, sel in selections.items()
        },
        "thirds": {
            label: selected_thirds(dirs[label], design.ctxs, k) for label in r100
        },
    }


def run_analyze(configs: list[dict[str, Any]], report_dir: Path = REPORT_DIR) -> Path:
    """Analyze exactly both datasets; write ``analysis.json`` and ``results.tex``."""
    if sorted(c["dataset"]["name"] for c in configs) != sorted(DATASETS):
        raise ValueError("analysis needs both datasets: bracs and tcga_ut")
    analysis = {
        "datasets": {
            c["dataset"]["name"]: analyze_dataset(c, report_dir) for c in configs
        }
    }
    write_json(report_dir / "analysis.json", analysis)
    (report_dir / "results.tex").write_text(render_results(analysis), encoding="utf-8")
    return report_dir / "analysis.json"
