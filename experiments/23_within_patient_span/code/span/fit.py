"""Fit stage: cohort basis plus within-patient complement arms of one (split, draw) shard.

Wa..Wd whiten with ``Sigma_B + s * tau * P Sigma_W P`` (P projects off the cohort basis), adding trace
``s * tr(Sigma_B)`` along the cohort's own within-patient directions; Ra..Rd keep those eigenvalues on
random complement directions. Wt/Rt fit nothing: they copy the record of the best-validating s, with
s = 0 being RWc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    ensure_dirs,
    read_run_record,
    split_paths,
    write_run_record,
)

from breadth.fit import EvalPartition, _build_draw_record, init_shard

from sites import allocation_dir

from centre import PATIENT_COUNTS, patches_per_patient
from centre.cohort import TrainingTable, draw_cohorts, training_table
from centre.fit import _tune_whitened, decode_shard_index, shard_count, split_arm
from centre.pool import Pool, load_pool

from directions import KAPPA_FACTORS
from directions.basis import cohort_eigenbasis

from spectrum.basis import subspace_overlap
from spectrum.fit import _best_score, select_beta

from span import ARMS, CAPTURE_MULTIPLES, S_GRID, S_VALUES, baseline_config
from span.basis import captured_top_k, complement_within, expanded, random_complement

__all__ = ["decode_shard_index", "shard_count", "run_fit_shard"]

# (cohort basis, cohort eigenvalues, complement basis, complement eigenvalues, random complement basis)
Context = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _context(
    table: TrainingTable, n_classes: int, g: int, seed: tuple[int, int]
) -> Context:
    """Cohort and complement bases of one patient count, with random directions seeded per (split, draw, g)."""
    u_b, lam_b = cohort_eigenbasis(table, n_classes, g)
    v, mu = complement_within(table, n_classes, g, u_b)
    rng = np.random.default_rng([*seed, g])
    return u_b, lam_b, v, mu, random_complement(u_b, len(mu), rng)


def _arm_pool(
    family: str, ctx: Context, pool: Pool
) -> tuple[Pool, dict[str, Any], float]:
    """Pool view carrying the arm's expanded basis, its diagnostics, and the kappa-anchoring scale."""
    u_b, lam_b, v, mu, v_random = ctx
    s = S_GRID[family[1:]]
    basis, eigvals, tau = expanded(
        u_b, lam_b, v_random if family[0] == "R" else v, mu, s
    )
    r_b = len(lam_b)
    extra = {
        "s": s,
        "tau": tau,
        "basis_rank": r_b,
        "added_rank": len(mu),
        "added_pool_overlap": subspace_overlap(
            basis[r_b : 2 * r_b], pool.b_basis[:r_b]
        ),
        "captured_cohort": captured_top_k(u_b, lam_b, pool, r_b),
        "captured_top_k": {
            str(m): captured_top_k(basis, eigvals, pool, m * r_b)
            for m in CAPTURE_MULTIPLES
        },
    }
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
    family, g = split_arm(arm)
    arm_pool, extra, anchor = _arm_pool(family, ctx, pool)
    factors = tuple(f * anchor for f in KAPPA_FACTORS)
    out, kappa = _tune_whitened(table.x, table.y, evals, arm_pool, factors)
    fit, lam, test_preds, test_probs, val_end, test_end = out
    meta = (g, patches_per_patient(g), draw_idx)
    rec = _build_draw_record(
        config,
        meta,
        lam,
        fit,
        (test_preds, test_probs, val_end, test_end),
        evals.test_y,
    )
    write_run_record(out_dir, {**rec, "arm": arm, **extra, **kappa}, keep_arrays=True)


def _fit_tuned(
    config: dict[str, Any],
    paths: dict[str, Path],
    kind: str,
    g: int,
    split_idx: int,
    draw_idx: int,
) -> None:
    """Copy the best-validating s's record (RWc for s = 0) as this shard's Wt/Rt record."""
    reused = split_paths(
        ensure_dirs(baseline_config(config, "whitening_outputs")), split_idx
    )
    sources = {
        s: allocation_dir(paths, f"{kind}{letter}{g}", draw_idx)
        for letter, s in S_GRID.items()
    } | {0.0: allocation_dir(reused, f"RWc{g}", draw_idx)}
    scores = {str(s): _best_score(sources[s]) for s in S_VALUES}
    s = select_beta(scores)
    record = read_run_record(sources[s])
    assert record is not None
    if (
        s == 0.0
    ):  # RWc's record carries no diagnostics: its added share is zero by definition.
        first = read_run_record(sources[S_VALUES[1]], array_fields=())
        assert first is not None
        cohort = first["captured_cohort"]
        record |= {
            "captured_cohort": cohort,
            "captured_top_k": {str(m): cohort for m in CAPTURE_MULTIPLES},
        }
    write_run_record(
        allocation_dir(paths, f"{kind}t{g}", draw_idx),
        {**record, "arm": f"{kind}t{g}", "s": s, "s_validation_scores": scores},
        keep_arrays=True,
    )


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending W/R arm of one (split, draw); Wt/Rt are copied last."""
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
    tables = {
        g: training_table(train_df, names, cohorts.nested, g) for g in PATIENT_COUNTS
    }
    pool = load_pool(baseline_config(config, "baseline_outputs"), split_idx, names)
    contexts: dict[int, Context] = {}
    for arm in pending:
        family, g = split_arm(arm)
        if family[1:] == "t":
            _fit_tuned(config, paths, family[0], g, split_idx, draw_idx)
            continue
        if g not in contexts:
            contexts[g] = _context(tables[g], len(names), g, (split_idx, draw_idx))
        out_dir = allocation_dir(paths, arm, draw_idx)
        _fit_arm(config, out_dir, arm, tables[g], contexts[g], pool, evals, draw_idx)
