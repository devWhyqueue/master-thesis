"""Precheck stage (0 fits): gates 0-2 (PLAN.md "Gate 0/1/2"). Gate 0 reads Step 1's fixed-lambda
prior-check JSON (shared by both datasets, produced outside this pipeline). Gates 1-2 read this
dataset's own B pools across every (split, draw) shard, draws 2-9 -- geometry only, no fits.

Every dataset always writes its own ``coverage_curve.json``, even when its own gates are skipped
(BRACS-only claims, PLAN.md "Gate 0"/"Gate 1"), so the other side's precheck can read it as a peer
file (``slurm.peer_outputs``, the same convention exp-36 uses for its own cross-dataset primary).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from breadth.fit import init_shard

from prevalence.fit import _patient_counts

from joint.fitting.shard import dataset_shard

from support import (
    GATE0_SUPPORT_GAP_PP,
    GATE1_NS,
    GATE1_RATIO,
    GATE1_TAIL_N,
    GATE2_COVERAGE_MAX_RATIO,
    GATE2_REDUNDANT_MIN_RATIO,
    MAIN_DRAWS,
    N_SPLITS,
)
from support.coverage import centre_error, coverage_deficit, patient_balanced_mean
from support.selection import greedy_quota_coverage, prefix_indices, redundant_pick
from support.shard import ClassPool, class_pools

__all__ = ["run_precheck"]

logger = logging.getLogger(__name__)


def _coverage_curve(config: dict[str, Any]) -> dict[str, list[float]]:
    """Per split, this dataset's own draw-pooled c(n) at every ``GATE1_NS``, on its tail class."""
    dataset = config["dataset"]["name"]
    curve: dict[str, list[float]] = {str(n): [] for n in GATE1_NS}
    for split_idx in range(N_SPLITS):
        train_df, names, _evals, _paths = init_shard(config, split_idx)
        per_n: dict[int, list[float]] = {n: [] for n in GATE1_NS}
        for draw_idx in MAIN_DRAWS:
            shard = dataset_shard(dataset, train_df, names, split_idx, draw_idx)
            pools = class_pools(shard)
            tail = min(pools, key=lambda p: p.s_count)
            for n in GATE1_NS:
                quota_n = _patient_counts(min(n, len(tail.x)), shard.g)
                idx = prefix_indices(tail.patient_idx, quota_n)
                per_n[n].append(coverage_deficit(tail.x, idx))
        for n in GATE1_NS:
            curve[str(n)].append(float(np.mean(per_n[n])))
    return curve


def gate0_support_gap(config: dict[str, Any]) -> dict[str, Any]:
    """BRACS only: matched native D_S(BRACS) - D_S(TCGA) at fixed lambda >= 1 pp (Step 1 data)."""
    if config["dataset"]["name"] != "bracs":
        return {"pass": True, "skipped": True, "reason": "gate 0 is BRACS-only"}
    path = Path(config["slurm"]["fixed_lambda_prior_check"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    d_s_bracs = float(payload["bracs"]["native_D_S_fixed_lambda_pp"])
    d_s_tcga = float(payload["tcga_ut"]["native_D_S_fixed_lambda_pp"])
    gap = d_s_bracs - d_s_tcga
    return {
        "d_s_bracs_pp": d_s_bracs,
        "d_s_tcga_ut_pp": d_s_tcga,
        "gap_pp": gap,
        "pass": bool(gap >= GATE0_SUPPORT_GAP_PP),
    }


def gate1_coverage_gap(
    config: dict[str, Any], curve: dict[str, list[float]]
) -> dict[str, Any]:
    """BRACS only: BRACS c(12) / TCGA-UT c(12) >= 1.2 in every split."""
    if config["dataset"]["name"] != "bracs":
        return {"pass": True, "skipped": True, "reason": "gate 1 is BRACS-only"}
    peer_path = Path(config["slurm"]["peer_outputs"]) / "data" / "coverage_curve.json"
    if not peer_path.exists():
        return {"pass": False, "reason": f"peer coverage curve not ready: {peer_path}"}
    peer_curve = json.loads(peer_path.read_text(encoding="utf-8"))
    key = str(GATE1_TAIL_N)
    ratios = [b / t for b, t in zip(curve[key], peer_curve[key])]
    return {"ratios": ratios, "pass": bool(all(r >= GATE1_RATIO for r in ratios))}


def _accumulate_pool(
    pool: ClassPool,
    coverage_ratios: list[float],
    redundant_ratios: list[float],
    eps_by_arm: dict[str, list[float]],
) -> None:
    """One thinned class's ratios and centre errors, folded into the running accumulators."""
    centre = patient_balanced_mean(pool.x, pool.patient_idx)
    picks = {
        "random": prefix_indices(pool.patient_idx, pool.quota),
        "coverage": greedy_quota_coverage(pool.x, pool.patient_idx, pool.quota),
        "redundant": redundant_pick(pool.x, pool.patient_idx, pool.quota, centre),
    }
    c = {arm: coverage_deficit(pool.x, idx) for arm, idx in picks.items()}
    if c["random"] > 0:
        coverage_ratios.append(c["coverage"] / c["random"])
        redundant_ratios.append(c["redundant"] / c["random"])
    for arm, idx in picks.items():
        eps_by_arm[arm].append(centre_error(pool.x, idx, centre))


def gate2_manipulation(config: dict[str, Any]) -> dict[str, Any]:
    """Both datasets: on thinned classes, coverage arm covers better, redundant arm covers worse."""
    dataset = config["dataset"]["name"]
    coverage_ratios: list[float] = []
    redundant_ratios: list[float] = []
    eps_by_arm: dict[str, list[float]] = {"random": [], "coverage": [], "redundant": []}
    for split_idx in range(N_SPLITS):
        train_df, names, _evals, _paths = init_shard(config, split_idx)
        for draw_idx in MAIN_DRAWS:
            shard = dataset_shard(dataset, train_df, names, split_idx, draw_idx)
            for pool in class_pools(shard):
                if pool.thinned:
                    _accumulate_pool(
                        pool, coverage_ratios, redundant_ratios, eps_by_arm
                    )
    passed = bool(
        np.mean(coverage_ratios) <= GATE2_COVERAGE_MAX_RATIO
        and np.mean(redundant_ratios) >= GATE2_REDUNDANT_MIN_RATIO
    )
    return {
        "coverage_ratio_mean": float(np.mean(coverage_ratios)),
        "redundant_ratio_mean": float(np.mean(redundant_ratios)),
        "epsilon_mean": {arm: float(np.mean(v)) for arm, v in eps_by_arm.items()},
        "pass": passed,
    }


def run_precheck(config: dict[str, Any]) -> Path:
    """Write this dataset's coverage curve, then run gates 0-2 and stop main submission on failure."""
    dataset = config["dataset"]["name"]
    curve = _coverage_curve(config)
    out_dir = output_root(config) / "data"
    write_json(out_dir / "coverage_curve.json", cast(Any, curve))
    results = {
        "support_gap": gate0_support_gap(config),
        "coverage_gap": gate1_coverage_gap(config, curve),
        "manipulation": gate2_manipulation(config),
    }
    overall = all(g["pass"] for g in results.values())
    payload = {"gates": results, "overall_pass": overall}
    diag_path = out_dir / "diagnostics.json"
    write_json(diag_path, payload)
    if not overall:
        failed = [name for name, g in results.items() if not g["pass"]]
        raise RuntimeError(f"Precheck gate(s) failed: {failed}; see {diag_path}")
    logger.info("All precheck gates passed for %s.", dataset)
    return diag_path
