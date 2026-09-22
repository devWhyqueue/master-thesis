"""Fit stage: U/Ut/It arms of one (split, draw) shard, cohort-only centre-uncertainty logistic regression.

Per patient count, every (covariance, t, lambda) candidate is fit and scored on validation only.
The selection is frozen to ``selection.json`` (fingerprinted, with the winning coefficients in
``winners.npz``) before any test scoring; run records are then written from the frozen winners.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
from decodability.linear import predict_logreg
from imbalance_benchmark.analysis.aggregation.parallel_cache import worker_count
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _macro_recall_mean,
)
from joblib import Parallel, delayed, parallel_config

from breadth.fit import EvalPartition, init_shard

from sites import allocation_dir

from centre import PATIENT_COUNTS
from centre.cohort import TrainingTable, draw_cohorts, training_table
from centre.fit import decode_shard_index, shard_count

from directions.basis import cohort_eigenbasis

from uncertainty import (
    COVARIANCE_KINDS,
    ISOTROPIC_EXTRA_T,
    LAMBDAS,
    MAX_ITER,
    NONZERO_T_FACTORS,
    TOLERANCE,
    baseline_arm_score,
)
from uncertainty.freeze import (
    WINNERS_NAME,
    Candidate,
    fingerprint,
    read_selection,
    t_grid,
    select_arms,
    write_selection,
)
from uncertainty.records import write_arm_records
from uncertainty.loss import (
    Covariance,
    UncertainFit,
    covariance_for,
    fit_uncertain_logistic,
)

__all__ = [
    "decode_shard_index",
    "shard_count",
    "run_fit_shard",
]

logger = logging.getLogger(__name__)


def _score(fit: UncertainFit, evals: EvalPartition) -> float:
    """Validation patient-macro recall of a fit."""
    preds, _ = predict_logreg(evals.val_x, fit.coef, fit.intercept)
    cases = evals.val_id["case_id"].astype(str).to_numpy()
    return float(_macro_recall_mean(evals.val_y, preds, cases))


def _fit_grid(
    table: TrainingTable,
    n_classes: int,
    g: int,
    evals: EvalPartition,
    t_values: tuple[float, ...] = NONZERO_T_FACTORS,
    kinds: tuple[str, ...] = COVARIANCE_KINDS,
) -> tuple[
    list[Candidate], dict[tuple[str, float, float], UncertainFit], dict[str, Any]
]:
    """Fit every nonzero-t candidate of the given covariance kinds; return scores, fits, and cohort geometry."""
    basis, eigvals = cohort_eigenbasis(table, n_classes, g)
    covs: dict[str, Covariance] = {
        k: covariance_for(k, basis, eigvals, g) for k in COVARIANCE_KINDS
    }
    grid = [(k, t, lam) for k in kinds for t in t_values for lam in LAMBDAS]
    with parallel_config(backend="loky", inner_max_num_threads=2):
        fits = cast(
            list[UncertainFit],
            Parallel(n_jobs=min(worker_count(), len(grid)))(
                delayed(fit_uncertain_logistic)(
                    table.x, table.y, n_classes, lam, t, covs[k], (TOLERANCE, MAX_ITER)
                )
                for k, t, lam in grid
            ),
        )
    candidates, by_key = [], {}
    for (k, t, lam), fit in zip(grid, fits):
        diag = {
            "objective": fit.objective,
            "grad_norm": fit.grad_norm,
            "n_iter": fit.n_iter,
            "converged": fit.converged,
        }
        candidates.append(Candidate(k, t, lam, _score(fit, evals), diag))
        by_key[(k, t, lam)] = fit
    geometry = {
        "rank": int(len(eigvals)),
        "trace": float(eigvals.sum()) / g,
        "zero_covariance": covs["patient"].is_zero,
    }
    return candidates, by_key, geometry


def _extend_selection(
    table: TrainingTable,
    n_classes: int,
    evals: EvalPartition,
    old: dict[str, Any],
    missing: tuple[float, ...],
    where: tuple[Path, str, int, float],
) -> dict[str, dict[str, Any] | None]:
    """Fit only the isotropic ``missing`` strengths, merge with the frozen candidates, and refreeze the selection."""
    sel_dir, fp, g, r_score = where
    new, by_key, geometry = _fit_grid(
        table, n_classes, g, evals, missing, ("isotropic",)
    )
    with np.load(sel_dir / WINNERS_NAME) as npz:
        for fam, sel in old["selected"].items():
            if sel is not None:
                key = (sel["kind"], sel["t"], sel["lam"])
                by_key[key] = UncertainFit(
                    npz[f"{fam}_coef"], npz[f"{fam}_intercept"], 0.0, 0.0, 0, True
                )
    merged = [Candidate(**c) for c in old["candidates"]] + new
    selected = select_arms(merged, r_score)
    write_selection(sel_dir, fp, merged, selected, geometry, r_score, by_key)
    return selected


def _fit_patient_count(
    config: dict[str, Any],
    paths: dict[str, Path],
    table: TrainingTable,
    n_classes: int,
    evals: EvalPartition,
    where: tuple[int, int, int],
) -> None:
    """Freeze (or extend, or verify) the selection of one patient count, then write its arm records."""
    split_idx, draw_idx, g = where
    r_score = baseline_arm_score(config, split_idx, draw_idx, f"R{g}")
    source = {"r_validation_score": r_score}
    fp_for = lambda grid: fingerprint(config, table, g, source, grid)  # noqa: E731
    sel_dir = allocation_dir(paths, f"Sel{g}", draw_idx)
    record = read_selection(sel_dir, fp_for)
    if record is None:
        candidates, by_key, geometry = _fit_grid(table, n_classes, g, evals)
        selected = select_arms(candidates, r_score)
        fp = fp_for((0.0, *NONZERO_T_FACTORS))
        write_selection(sel_dir, fp, candidates, selected, geometry, r_score, by_key)
        record = read_selection(sel_dir, fp_for)
        assert record is not None
    full_grid = (0.0, *NONZERO_T_FACTORS, *ISOTROPIC_EXTRA_T)
    if t_grid(record) != full_grid:
        missing = tuple(t for t in ISOTROPIC_EXTRA_T if t not in t_grid(record))
        selected = _extend_selection(
            table,
            n_classes,
            evals,
            record,
            missing,
            (sel_dir, fp_for(full_grid), g, r_score),
        )
        for fam in selected:  # a changed winner invalidates its scored run record
            if selected[fam] != record["selected"][fam]:
                shutil.rmtree(allocation_dir(paths, f"{fam}{g}", draw_idx), True)
    record = read_selection(sel_dir, fp_for)
    assert record is not None
    write_arm_records(config, paths, record, sel_dir, evals, len(table.y), where)


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit and freeze selections, then write every missing U/Ut/It record of one (split, draw) shard."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    cohorts = draw_cohorts(train_df, names, split_idx, draw_idx)
    for g in PATIENT_COUNTS:
        table = training_table(train_df, names, cohorts.nested, g)
        logger.info("Split %d, draw %d, G=%d", split_idx, draw_idx, g)
        _fit_patient_count(
            config, paths, table, len(names), evals, (split_idx, draw_idx, g)
        )
