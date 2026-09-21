"""Fit stage: oracle-weighted complement arms (Wo, Ro, Po) of one (split, draw) shard at G = 5.

All arms keep the cohort block (U, lambda) and add a complement block whose eigenvalues are the pool's
between-patient variance along each direction: Wo on the cohort's within-patient directions V, Ro on
random complement directions, Po on the top eigenvectors of the pool covariance projected off U (with
their own eigenvalues). The pool is oracle information, so this is a mechanism study, not a remedy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import RUN_RECORD_NAME, write_run_record
from scipy.stats import rankdata

from breadth.fit import EvalPartition, _build_draw_record, init_shard

from sites import allocation_dir

from centre import patches_per_patient
from centre.cohort import TrainingTable, draw_cohorts, training_table
from centre.fit import _tune_whitened, decode_shard_index, shard_count, split_arm
from centre.pool import Pool, load_pool

from directions import KAPPA_FACTORS

from span.fit import Context, _context

from weighting import ARMS, G, baseline_config
from weighting.basis import complement_pool, oracle_expanded

__all__ = ["decode_shard_index", "shard_count", "run_fit_shard"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two equal-length vectors."""
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _arm_pool(
    family: str, ctx: Context, pool: Pool
) -> tuple[Pool, dict[str, Any], float]:
    """Pool view carrying the arm's oracle-weighted basis, its diagnostics, and the kappa-anchoring scale."""
    u_b, lam_b, v, mu, v_random = ctx
    extra: dict[str, Any] = {"added_rank": len(mu)}
    if family == "Po":
        directions, weights = complement_pool(u_b, pool)
        basis, eigvals = oracle_expanded(u_b, lam_b, directions, pool, weights)
        extra["added_rank"] = len(weights)
    else:
        directions = v if family == "Wo" else v_random
        basis, eigvals = oracle_expanded(u_b, lam_b, directions, pool)
        weights = eigvals[len(lam_b) :]
    extra["oracle_implied_s"] = float(weights.sum() / lam_b.sum())
    extra["added_pool_share"] = float(weights.sum() / pool.b_eigvals.sum())
    if family == "Wo":
        extra["spearman_omega_oracle"] = _spearman(mu, weights)
    # kappa = factor / mean(eigvals) would whiten the cohort block harder once eigenvalues are added.
    anchor = float(eigvals.mean() / lam_b.mean())
    return pool._replace(b_basis=basis, b_eigvals=eigvals), extra, anchor


def _fit_arm(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    table: TrainingTable,
    ctx: Context,
    pool: Pool,
    evals: EvalPartition,
    draw_idx: int,
) -> None:
    """Tune kappa and lambda for one arm's whitening and write its run record."""
    family, _ = split_arm(arm)
    arm_pool, extra, anchor = _arm_pool(family, ctx, pool)
    factors = tuple(f * anchor for f in KAPPA_FACTORS)
    out, kappa = _tune_whitened(table.x, table.y, evals, arm_pool, factors)
    fit, lam, test_preds, test_probs, val_end, test_end = out
    meta = (G, patches_per_patient(G), draw_idx)
    rec = _build_draw_record(
        config,
        meta,
        lam,
        fit,
        (test_preds, test_probs, val_end, test_end),
        evals.test_y,
    )
    write_run_record(out_dir, {**rec, "arm": arm, **extra, **kappa}, keep_arrays=True)


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending arm of one (split, draw)."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    pending = [
        a
        for a in ARMS
        if not (allocation_dir(paths, a, draw_idx) / RUN_RECORD_NAME).exists()
    ]
    if not pending:
        return
    cohorts = draw_cohorts(train_df, names, split_idx, draw_idx)
    table = training_table(train_df, names, cohorts.nested, G)
    pool = load_pool(baseline_config(config, "baseline_outputs"), split_idx, names)
    ctx = _context(table, len(names), G, (split_idx, draw_idx))
    for arm in pending:
        out_dir = allocation_dir(paths, arm, draw_idx)
        _fit_arm(config, out_dir, arm, table, ctx, pool, evals, draw_idx)
