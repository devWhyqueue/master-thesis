"""Fit stage: cohort-basis spectrum arms of one (split, draw) shard, reusing exp-16/18 cohorts and pool.

F/Ba/Bb/Bc keep the cohort eigenbasis and compress its eigenvalues to ``lambda ** beta``; O sets
them to the pool's variance along each cohort direction; P swaps the basis for the top within-patient
PCs. Bt fits nothing: it copies the record of the best-validating beta, with beta = 1 being RWc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    ensure_dirs,
    read_run_record,
    split_paths,
    write_run_record,
)

from breadth import TIE_TOLERANCE
from breadth.fit import EvalPartition, _build_draw_record, init_shard

from sites import allocation_dir

from centre import PATIENT_COUNTS, patches_per_patient
from centre.cohort import TrainingTable, draw_cohorts, training_table
from centre.fit import _tune_whitened, decode_shard_index, shard_count, split_arm
from centre.pool import Pool, load_pool

from directions import KAPPA_FACTORS
from directions.basis import cohort_eigenbasis

from spectrum import ARMS, BETAS, baseline_config
from spectrum.basis import (
    oracle_eigvals,
    oracle_slope,
    pool_variance_captured,
    spectrum_eigvals,
    subspace_overlap,
    within_patient_basis,
)

__all__ = ["decode_shard_index", "shard_count", "select_beta", "run_fit_shard"]

# Bt's candidate betas in ascending order, with the reused RWc arm as beta = 1.
_BT_BETAS: tuple[float, ...] = (*BETAS.values(), 1.0)


def select_beta(scores: dict[str, float]) -> float:
    """Beta ``Bt`` picks from these validation scores (ties to the larger beta)."""
    best, best_score = -1.0, -1.0
    for beta in sorted(scores, key=float):
        if scores[beta] >= best_score - TIE_TOLERANCE:
            best, best_score = float(beta), scores[beta]
    return best


def _arm_pool(
    family: str, table: TrainingTable, n_classes: int, g: int, pool: Pool
) -> tuple[Pool, dict[str, Any]]:
    """Pool view carrying this arm's basis and eigenvalues, plus its diagnostics."""
    basis, eigvals = cohort_eigenbasis(table, n_classes, g)
    extra: dict[str, Any] = {"basis_rank": len(eigvals)}
    if family in BETAS:
        extra["beta"] = BETAS[family]
        eigvals = spectrum_eigvals(eigvals, BETAS[family])
    elif family == "O":
        oracle = oracle_eigvals(basis, pool)
        extra["oracle_slope"] = oracle_slope(eigvals, oracle)
        extra["pool_variance_captured"] = pool_variance_captured(basis, pool)
        eigvals = oracle
    else:
        p_basis = within_patient_basis(table, n_classes, g, len(eigvals))
        extra["subspace_overlap"] = subspace_overlap(basis, p_basis)
        basis, eigvals = p_basis, eigvals[: len(p_basis)]
    return pool._replace(b_basis=basis, b_eigvals=eigvals), extra


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
    """Tune kappa and lambda for one arm's whitening and write its run record."""
    family, g = split_arm(arm)
    arm_pool, extra = _arm_pool(family, table, n_classes, g, pool)
    out, kappa = _tune_whitened(table.x, table.y, evals, arm_pool, KAPPA_FACTORS)
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


def _best_score(out_dir: Path) -> float:
    """Validation score of the kappa an arm's record selected."""
    record = read_run_record(out_dir, array_fields=())
    assert record is not None, f"missing record {out_dir}"
    return float(record["kappa_validation_scores"][str(record["kappa_factor"])])


def _fit_bt(
    config: dict[str, Any],
    paths: dict[str, Path],
    g: int,
    split_idx: int,
    draw_idx: int,
) -> None:
    """Copy the best-validating beta's record (RWc for beta = 1) as this shard's Bt record."""
    reused = split_paths(
        ensure_dirs(baseline_config(config, "whitening_outputs")), split_idx
    )
    sources = {
        beta: allocation_dir(paths, f"{family}{g}", draw_idx)
        for family, beta in BETAS.items()
    } | {1.0: allocation_dir(reused, f"RWc{g}", draw_idx)}
    scores = {str(beta): _best_score(sources[beta]) for beta in _BT_BETAS}
    beta = select_beta(scores)
    record = read_run_record(sources[beta])
    assert record is not None
    write_run_record(
        allocation_dir(paths, f"Bt{g}", draw_idx),
        {**record, "arm": f"Bt{g}", "beta": beta, "beta_validation_scores": scores},
        keep_arrays=True,
    )


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending spectrum arm of one (split, draw); Bt is copied last."""
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
    for arm in pending:
        family, g = split_arm(arm)
        if family == "Bt":
            _fit_bt(config, paths, g, split_idx, draw_idx)
        else:
            out_dir = allocation_dir(paths, arm, draw_idx)
            _fit_arm(config, out_dir, arm, tables[g], len(names), pool, evals, draw_idx)
