"""Six paired without-LoRA versus with-LoRA accuracy contrasts and the joint omission verdict.

Candidate strengths are selected per representation from validation records only (exp-40's
``analyze.select``, at each method's own stage). Test balanced-accuracy distributions come from
the shared patient-clustered bootstrap (10,000 replicates, one set of draw weights), so each
contrast distribution is the difference of two paired distributions; intervals use the replicate
columns only and column zero holds the observed estimate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.common import output_root

from analyze.bootstrap import selected_ba, selected_thirds
from analyze.paths import Shard
from analyze.paths import run_dir as shard_run_dir
from analyze.select import select, selected_frequency

from breadth import BOOTSTRAP_SEED, exp2_split_paths
from breadth.analyze.canonical import canonical_class_names, canonical_permutation

from centre import N_DRAWS, N_SPLITS
from centre.analyze import pooled

from decomposition.model import draw_weights

from lora_ablation import ARM, MARGIN_PP, METHODS, N_CONTRASTS, STAGE_OF
from lora_ablation.inputs import (
    adapted_paths,
    check_inventory,
    frozen_paths,
    summarize_resources,
)
from lora_ablation.preflight import INVENTORY_NAME
from lora_ablation.render import render_results

__all__ = [
    "METHODS",
    "summarize_contrast",
    "joint_verdict",
    "select_candidates",
    "analyze_dataset",
    "analyze_experiment",
    "run_analyze",
]

N_REPLICATES = 10_000
ALPHA = 0.05 / N_CONTRASTS  # Bonferroni over the six primary contrasts
PRIMARY_QUANTILES = (ALPHA / 2, 1 - ALPHA / 2)
DESCRIPTIVE_QUANTILES = (0.025, 0.975)
DATASETS = ("bracs", "tcga_ut")
REPORT_DIR = Path(__file__).resolve().parents[2] / "report"


def _interval(
    distribution: np.ndarray, quantiles: tuple[float, float]
) -> dict[str, float]:
    lower, upper = (float(q) for q in np.quantile(distribution[1:], quantiles))
    return {"estimate": float(distribution[0]), "lower": lower, "upper": upper}


def summarize_contrast(distribution: np.ndarray) -> dict[str, float | str]:
    """Interval and noninferiority decision of a contrast (observed estimate at index zero)."""
    distribution = np.asarray(distribution, dtype=np.float64)
    if distribution.ndim != 1 or distribution.size < 2:
        raise ValueError(
            "a contrast distribution is one-dimensional: estimate, then replicates"
        )
    result = _interval(distribution, PRIMARY_QUANTILES)
    if result["lower"] > -MARGIN_PP:
        decision = "noninferior"
    elif result["upper"] < -MARGIN_PP:
        decision = "lora_advantage"
    else:
        decision = "inconclusive"
    return {**result, "decision": decision}


def joint_verdict(decisions: dict[str, str]) -> str:
    """Omitting LoRA is supported in the tested scope only if all six contrasts are noninferior."""
    if set(decisions) != {f"{d}/{m}" for d in DATASETS for m in METHODS}:
        raise ValueError("the joint verdict needs exactly the six primary contrasts")
    ok = all(v == "noninferior" for v in decisions.values())
    return "omission_supported" if ok else "not_supported"


def select_candidates(
    config: dict[str, Any], paths: dict[int, dict[str, Path]], arm: str, method: str
) -> dict[Shard, float | None]:
    """Per shard, the candidate with the best validation BA (test is never opened; ties: config order)."""
    return select(config, paths, arm, STAGE_OF[method], method)


class _Design:
    """Shared bootstrap contexts, class order, and draw weights of one dataset."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.names = canonical_class_names(config)
        self.ctxs = {
            s: BootstrapContext(
                exp2_split_paths(config, s),
                is_mil=False,
                n_replicates=N_REPLICATES,
                seed=BOOTSTRAP_SEED,
            )
            for s in range(N_SPLITS)
        }
        self.perms = {
            s: canonical_permutation(config, s, self.names) for s in range(N_SPLITS)
        }
        fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        self.weights = draw_weights(fit_split, N_DRAWS, self.ctxs[0].n_replicates, rng)


def _method_block(
    config: dict[str, Any],
    design: _Design,
    paths: dict[int, dict[str, Path]],
    method: str,
) -> dict[str, Any]:
    selection = select_candidates(config, paths, ARM, method)
    dirs = {
        (s, d): shard_run_dir(paths[s], ARM, method, param, d)
        for (s, d), param in selection.items()
    }
    ba = selected_ba(dirs, design.ctxs, design.perms, len(design.names))
    return {
        "ba": pooled(ba, design.weights),
        "frequency": selected_frequency(selection),
        "thirds": selected_thirds(dirs, design.ctxs, len(design.names)),
    }


def _representation(
    config: dict[str, Any], design: _Design, paths: dict[int, dict[str, Path]]
) -> dict[str, dict[str, Any]]:
    return {m: _method_block(config, design, paths, m) for m in ("ce", *METHODS)}


def _descriptive(frozen: dict, adapted: dict) -> dict[str, Any]:
    """Absolute BA, CE-relative gains, selections, tail recall: descriptive 95% intervals only."""
    reps = {"frozen": frozen, "adapted": adapted}
    interval = lambda d: _interval(d, DESCRIPTIVE_QUANTILES)  # noqa: E731
    return {
        "absolute_ba": {
            r: {m: interval(b["ba"]) for m, b in rep.items()} for r, rep in reps.items()
        },
        "ce_relative_gain": {
            r: {m: interval(rep[m]["ba"] - rep["ce"]["ba"]) for m in METHODS}
            for r, rep in reps.items()
        },
        "ce_control_delta": interval(frozen["ce"]["ba"] - adapted["ce"]["ba"]),
        "selection_frequency": {
            r: {m: b["frequency"] for m, b in rep.items()} for r, rep in reps.items()
        },
        "thirds": {
            r: {m: b["thirds"] for m, b in rep.items()} for r, rep in reps.items()
        },
    }


def analyze_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """All primary and descriptive results of one dataset; rejects unaudited or incomplete inputs."""
    inventory = check_inventory(config)
    design = _Design(config)
    frozen = _representation(config, design, frozen_paths(config))
    adapted = _representation(config, design, adapted_paths(config))
    return {
        "dataset": config["dataset"]["name"],
        "replicates": design.ctxs[0].n_replicates - 1,
        "inventory_fingerprint": inventory["fingerprint"],
        "workload": inventory["workload"],
        "primary": {
            m: summarize_contrast(frozen[m]["ba"] - adapted[m]["ba"]) for m in METHODS
        },
        "resources": summarize_resources(config),
        **_descriptive(frozen, adapted),
    }


def analyze_experiment(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze exactly both datasets and report their six contrasts and the joint verdict."""
    if sorted(c["dataset"]["name"] for c in configs) != sorted(DATASETS):
        raise ValueError("analysis needs both datasets: bracs and tcga_ut")
    datasets = {c["dataset"]["name"]: analyze_dataset(c) for c in configs}
    decisions = {
        f"{d}/{m}": datasets[d]["primary"][m]["decision"]
        for d in DATASETS
        for m in METHODS
    }
    return {
        "protocol": "strong-mitigations-v2",
        "margin_pp": MARGIN_PP,
        "quantiles": list(PRIMARY_QUANTILES),
        "interval": "99.17% percentile (Bonferroni over six), replicate columns only",
        "datasets": datasets,
        "decisions": decisions,
        "joint_verdict": joint_verdict(decisions),
    }


def run_analyze(configs: list[dict[str, Any]]) -> Path:
    """Write ``report/analysis.json``, inventory copies, and ``results.tex``."""
    analysis = analyze_experiment(configs)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "analysis.json"
    path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "results.tex").write_text(render_results(analysis), encoding="utf-8")
    for config in configs:
        inventory = output_root(config) / INVENTORY_NAME
        name = f"inventory_{config['dataset']['name']}.json"
        (REPORT_DIR / name).write_text(
            inventory.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return path
