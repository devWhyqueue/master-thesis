"""Fit stage: U/Ut/It arms of one (split, draw) shard, cohort-only centre-uncertainty logistic regression.

Per patient count, every (covariance, t, lambda) candidate is fit and scored on validation only.
The selection is frozen to ``selection.json`` (fingerprinted, with the winning coefficients in
``winners.npz``) before any test scoring; run records are then written from the frozen winners.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from decodability.linear import predict_logreg
from imbalance_benchmark.analysis.aggregation.parallel_cache import worker_count
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _cluster_discrimination,
    _macro_recall_mean,
    clustered_endpoints,
)
from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    read_run_record,
    write_run_record,
)
from joblib import Parallel, delayed, parallel_config

from breadth.fit import EvalPartition, _build_draw_record, init_shard

from sites import allocation_dir

from centre import PATIENT_COUNTS, patches_per_patient
from centre.cohort import TrainingTable, draw_cohorts, training_table
from centre.fit import decode_shard_index, shard_count

from directions.basis import cohort_eigenbasis

from uncertainty import (
    ARMS,
    COVARIANCE_KINDS,
    LAMBDAS,
    MAX_ITER,
    NONZERO_T_FACTORS,
    TOLERANCE,
    baseline_arm_dir,
    baseline_arm_score,
)
from uncertainty.freeze import (
    WINNERS_NAME,
    Candidate,
    fingerprint,
    read_selection,
    select_arms,
    write_selection,
)
from uncertainty.loss import (
    Covariance,
    UncertainFit,
    as_linear_result,
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
    table: TrainingTable, n_classes: int, g: int, evals: EvalPartition
) -> tuple[
    list[Candidate], dict[tuple[str, float, float], UncertainFit], dict[str, Any]
]:
    """Fit every nonzero-t candidate of both covariance kinds; return scores, fits, and cohort geometry."""
    basis, eigvals = cohort_eigenbasis(table, n_classes, g)
    covs: dict[str, Covariance] = {
        k: covariance_for(k, basis, eigvals, g) for k in COVARIANCE_KINDS
    }
    grid = [
        (k, t, lam)
        for k in COVARIANCE_KINDS
        for t in NONZERO_T_FACTORS
        for lam in LAMBDAS
    ]
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


def _evaluate(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    meta: tuple[int, int, int],
    lam: float,
    fit: UncertainFit,
    evals: EvalPartition,
    extra: dict[str, Any],
) -> None:
    """Score a frozen winner on validation and test and write its run record."""
    g, n, draw_idx = meta
    result = as_linear_result(fit, lam, n, TOLERANCE, MAX_ITER)
    cases = evals.val_id["case_id"].astype(str).to_numpy()
    slides = evals.val_id["slide_id"].astype(str).to_numpy()
    v_preds, _ = predict_logreg(evals.val_x, fit.coef, fit.intercept)
    val_end = _cluster_discrimination(evals.val_y, v_preds, cases, slides, is_mil=False)
    preds, probs = predict_logreg(evals.test_x, fit.coef, fit.intercept)
    test_end = clustered_endpoints(
        evals.test_y, preds, probs, evals.test_id, is_mil=False
    )
    rec = _build_draw_record(
        config,
        (g, patches_per_patient(g), draw_idx),
        lam,
        result,
        (preds, probs, val_end, test_end),
        evals.test_y,
    )
    write_run_record(out_dir, {**rec, "arm": arm, **extra}, keep_arrays=True)


def _write_arm_records(
    config: dict[str, Any],
    paths: dict[str, Path],
    record: dict[str, Any],
    sel_dir: Path,
    evals: EvalPartition,
    n: int,
    where: tuple[int, int, int],
) -> None:
    """Write every missing arm run record of one patient count from the frozen selection."""
    split_idx, draw_idx, g = where
    with np.load(sel_dir / WINNERS_NAME) as npz:
        winners = {k: npz[k] for k in npz.files}
    for family, sel in record["selected"].items():
        arm = f"{family}{g}"
        out_dir = allocation_dir(paths, arm, draw_idx)
        if (out_dir / RUN_RECORD_NAME).exists():
            continue
        if sel is None:  # t = 0 wins: R's record is this arm's record.
            base = read_run_record(
                baseline_arm_dir(config, split_idx, draw_idx, f"R{g}")
            )
            assert base is not None
            extra = {"arm": arm, "t": 0.0, "covariance": None}
            write_run_record(out_dir, {**base, **extra}, keep_arrays=True)
            continue
        fit = UncertainFit(
            winners[f"{family}_coef"], winners[f"{family}_intercept"], 0.0, 0.0, 0, True
        )
        extra = {
            "t": sel["t"],
            "covariance": sel["kind"],
            "selection_fingerprint": record["fingerprint"],
        }
        _evaluate(config, out_dir, arm, (g, n, draw_idx), sel["lam"], fit, evals, extra)


def _fit_patient_count(
    config: dict[str, Any],
    paths: dict[str, Path],
    table: TrainingTable,
    n_classes: int,
    evals: EvalPartition,
    where: tuple[int, int, int],
) -> None:
    """Freeze (or verify) the selection of one patient count, then write its arm records."""
    split_idx, draw_idx, g = where
    r_score = baseline_arm_score(config, split_idx, draw_idx, f"R{g}")
    fp = fingerprint(config, table, g, {"r_validation_score": r_score})
    sel_dir = allocation_dir(paths, f"Sel{g}", draw_idx)
    record = read_selection(sel_dir, fp)
    if record is None:
        candidates, by_key, geometry = _fit_grid(table, n_classes, g, evals)
        selected = select_arms(candidates, r_score)
        write_selection(sel_dir, fp, candidates, selected, geometry, r_score, by_key)
        record = read_selection(sel_dir, fp)
        assert record is not None
    _write_arm_records(config, paths, record, sel_dir, evals, len(table.y), where)


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit and freeze selections, then write every missing U/Ut/It record of one (split, draw) shard."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    if all(
        (allocation_dir(paths, a, draw_idx) / RUN_RECORD_NAME).exists() for a in ARMS
    ):
        return
    cohorts = draw_cohorts(train_df, names, split_idx, draw_idx)
    for g in PATIENT_COUNTS:
        table = training_table(train_df, names, cohorts.nested, g)
        logger.info("Split %d, draw %d, G=%d", split_idx, draw_idx, g)
        _fit_patient_count(
            config, paths, table, len(names), evals, (split_idx, draw_idx, g)
        )
