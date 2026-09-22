"""Fit stage: whitened RW/RWc arms of one (split, draw) shard, reusing exp-16 cohorts and pool.

Records fitted on a smaller kappa grid are extended in place: only the missing factors are fit,
and the record is rewritten with the new fit only if one of them wins validation selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from breadth import TIE_TOLERANCE
from breadth.fit import EvalPartition, _build_draw_record, init_shard
from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    read_run_record,
    write_json,
    write_run_record,
)

from sites import allocation_dir

from centre import PATIENT_COUNTS, patches_per_patient
from centre.cohort import TrainingTable, draw_cohorts, training_table
from centre.fit import _tune_whitened, decode_shard_index, shard_count, split_arm
from centre.pool import Pool, load_pool

from directions import ARMS, KAPPA_FACTORS, exp16_config
from directions.basis import cohort_eigenbasis

__all__ = ["decode_shard_index", "shard_count", "select_kappa", "run_fit_shard"]


def select_kappa(scores: dict[str, float]) -> str:
    """Factor key ``_tune_whitened`` would pick from these validation scores (ties to larger kappa)."""
    best, best_score = "", -1.0
    for key in sorted(scores, key=float):
        if not best or scores[key] >= best_score - TIE_TOLERANCE:
            best, best_score = key, scores[key]
    return best


def _done_scores(out_dir: Path) -> dict[str, float]:
    """Stored kappa validation scores of an arm's run record, empty if not fitted yet."""
    record = read_run_record(out_dir, array_fields=())
    return {} if record is None else record["kappa_validation_scores"]


def _fit_arm(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    table: TrainingTable,
    n_classes: int,
    pool: Pool,
    evals: EvalPartition,
    draw_idx: int,
) -> None:
    old = read_run_record(out_dir, array_fields=())
    done = _done_scores(out_dir)
    missing = tuple(f for f in KAPPA_FACTORS if str(f) not in done)
    family, g = split_arm(arm)
    if family == "RWc":
        b_basis, b_eigvals = cohort_eigenbasis(table, n_classes, g)
        basis = pool._replace(b_basis=b_basis, b_eigvals=b_eigvals)
    else:
        basis = pool
    out, extra = _tune_whitened(table.x, table.y, evals, basis, missing)
    scores = {**done, **extra["kappa_validation_scores"]}
    if old is not None and select_kappa(scores) in done:
        write_json(
            out_dir / RUN_RECORD_NAME, {**old, "kappa_validation_scores": scores}
        )
        return
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
    write_run_record(
        out_dir,
        {
            **rec,
            "arm": arm,
            "basis": "cohort" if family == "RWc" else "pool",
            "basis_rank": len(basis.b_eigvals),
            **extra,
            "kappa_validation_scores": scores,
        },
        keep_arrays=True,
    )


def _shard_inputs(
    config: dict[str, Any],
    train_df: Any,
    names: list[str],
    split_idx: int,
    draw_idx: int,
) -> tuple[dict[int, TrainingTable], Pool]:
    """This shard's nested-cohort training tables per G and the shared exp-16 pool."""
    cohorts = draw_cohorts(train_df, names, split_idx, draw_idx)
    tables = {
        g: training_table(train_df, names, cohorts.nested, g) for g in PATIENT_COUNTS
    }
    return tables, load_pool(exp16_config(config), split_idx, names)


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every RW/RWc arm of one (split, draw) whose record lacks a factor of ``KAPPA_FACTORS``."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    pending = [
        a
        for a in ARMS
        if set(map(str, KAPPA_FACTORS))
        - set(_done_scores(allocation_dir(paths, a, draw_idx)))
    ]
    if not pending:
        return
    tables, pool = _shard_inputs(config, train_df, names, split_idx, draw_idx)
    for arm in pending:
        _, g = split_arm(arm)
        out_dir = allocation_dir(paths, arm, draw_idx)
        _fit_arm(config, out_dir, arm, tables[g], len(names), pool, evals, draw_idx)
