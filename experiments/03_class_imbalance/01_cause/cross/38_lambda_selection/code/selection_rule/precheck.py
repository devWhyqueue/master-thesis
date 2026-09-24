"""Precheck stage (0 refits): the phase-1 diagnostic. Re-predicts exp-36's own pilot-draw grids
(0-1) under every rule, verifies exact reproduction of exp-36's stored tuned/fixed D_P (integrity),
and -- BRACS only, once TCGA-UT's own phase-1 diagnostic is ready -- gates on cross-fit closure
(PLAN.md "Phase-1 gate"). Fail: short gate-failure report, no phase-2 fits (memory rule: gate
diagnostic before report).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths, write_json

from breadth.fit import init_shard

from selection_rule import INTEGRITY_TOL_PP, CLOSURE_GATE, N_SPLITS, PILOT_DRAWS
from selection_rule.analyze import (
    canonical_class_names,
    combine,
    gap_closure,
    pooled_ba,
)
from selection_rule.rules import run_rules

__all__ = ["run_precheck"]

logger = logging.getLogger(__name__)


def _stored_prior_check(config: dict[str, Any]) -> dict[str, float]:
    path = Path(config["slurm"]["fixed_lambda_prior_check"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, float], payload[config["dataset"]["name"]])


def _integrity(config: dict[str, Any], dists: dict[str, np.ndarray]) -> dict[str, Any]:
    """Re-derived tuned/fixed D_P must equal exp-36's own stored values (no refit happened)."""
    stored = _stored_prior_check(config)
    tuned_diff = abs(float(dists["D_P_tuned"][0]) - stored["native_D_P_tuned_pp"])
    fixed_diff = abs(
        float(dists["D_P_fixed"][0]) - stored["native_D_P_fixed_lambda_pp"]
    )
    return {
        "tuned_diff_pp": tuned_diff,
        "fixed_diff_pp": fixed_diff,
        "pass": bool(tuned_diff <= INTEGRITY_TOL_PP and fixed_diff <= INTEGRITY_TOL_PP),
    }


def _closure_gate(
    config: dict[str, Any], dists: dict[str, np.ndarray]
) -> dict[str, Any]:
    """BRACS only: cross-fit closure of the tuned D_P gap against TCGA-UT >= CLOSURE_GATE."""
    if config["dataset"]["name"] != "bracs":
        return {"pass": True, "skipped": True, "reason": "closure gate is BRACS-only"}
    peer_path = (
        Path(config["slurm"]["peer_outputs"]) / "data" / "phase1_distributions.npz"
    )
    if not peer_path.exists():
        return {
            "pass": False,
            "reason": f"peer phase-1 distributions not ready: {peer_path}",
        }
    with np.load(peer_path) as npz:
        peer = {k: np.asarray(npz[k]) for k in npz.files}
    cross_fit = gap_closure(dists, peer, "oracle")
    naive = gap_closure(dists, peer, "naive")
    return {
        "gap_tuned_pp": cross_fit["gap_tuned_estimate"],
        "delta_pp": cross_fit["delta_estimate"],
        "closure": cross_fit["closure"],
        "naive_closure": naive["closure"],
        "label": cross_fit["label"],
        "pass": bool(cross_fit["closure"] >= CLOSURE_GATE),
    }


def _run_pilot_rules(config: dict[str, Any]) -> None:
    """Every split's pilot-draw rules, source = exp-36's own stored native grids."""
    joint36_config = {
        **config,
        "paths": {"outputs": str(config["slurm"]["joint36_outputs"])},
    }
    for split_idx in range(N_SPLITS):
        _, _, evals, dest_paths = init_shard(config, split_idx)
        source_paths = split_paths(ensure_dirs(joint36_config), split_idx)
        for draw_idx in PILOT_DRAWS:
            run_rules(config, dest_paths, source_paths, draw_idx, evals)


def run_precheck(config: dict[str, Any]) -> Path:
    """Re-predict pilot-draw grids under every rule, then run the integrity and closure gates."""
    names = canonical_class_names(config)
    _run_pilot_rules(config)
    dists = combine(pooled_ba(config, names, PILOT_DRAWS))
    np.savez(
        output_root(config) / "data" / "phase1_distributions.npz", **cast(Any, dists)
    )

    results = {
        "integrity": _integrity(config, dists),
        "closure": _closure_gate(config, dists),
    }
    overall = all(g["pass"] for g in results.values())
    payload = {"gates": results, "overall_pass": overall}
    diag_path = output_root(config) / "report" / "phase1_diagnostic.json"
    write_json(diag_path, cast(Any, payload))
    if not overall:
        failed = [name for name, g in results.items() if not g["pass"]]
        raise RuntimeError(f"Precheck gate(s) failed: {failed}; see {diag_path}")
    logger.info("Phase-1 gates passed for %s.", config["dataset"]["name"])
    return diag_path
