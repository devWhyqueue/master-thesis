"""Adjacent-decade diagnostic refits for main fits that selected a grid endpoint.

Protocol phase 01: a main selection at 1e-8 or 1e2 gets one extra candidate one decade
beyond that endpoint. The extra candidate joins the stored frozen-grid scores under the
same selection rule; the frozen-grid primary analysis stays unchanged, and the refit
outcome is written separately to ``data/boundary_refits.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _macro_recall_mean,
)
from imbalance_benchmark.common import N_PATIENT_SPLITS, output_root, write_json
from imbalance_benchmark.datasets.features.cache import reset_feature_bank

from decodability.linear import fit_multinomial_logistic, predict_logreg

from prevalence import patients_per_class
from prevalence.fit import _shard_context

from transfer import ENCODERS, FIT_SOURCE, LAMBDAS, MAX_ITER, TIE_TOLERANCE, TOLERANCE
from transfer.analyze.common import fit_dirs, paths_by_split, require_record
from transfer.fit import init_shard
from transfer.fit.tuning import _allocate

__all__ = ["select_lambda", "run_boundary_refits"]

_EXTRA = {LAMBDAS[0]: LAMBDAS[0] / 10.0, LAMBDAS[-1]: LAMBDAS[-1] * 10.0}


def select_lambda(scores: dict[float, float | None]) -> float:
    """Frozen selection rule over any lambda set: ascending scan, ties to the stronger penalty."""
    best_lam, best_score = None, -1.0
    for lam in sorted(scores):
        score = scores[lam]
        if score is None:
            continue
        if best_lam is None or score - best_score >= -TIE_TOLERANCE:
            best_lam, best_score = lam, score
    if best_lam is None:
        raise RuntimeError("No converged candidate")
    return best_lam


def _cases(frame: pd.DataFrame) -> np.ndarray:
    return frame["case_id"].astype(str).to_numpy()


def _refit_cell(
    shard_inputs: tuple[Any, Any], record: dict[str, Any], arm: str
) -> dict[str, Any]:
    """Fit the extra candidate, rerun selection, and score the frozen and refit choices."""
    shard, evals = shard_inputs
    frozen = float(record["selected_lambda"])
    extra = _EXTRA[frozen]
    allocation = _allocate(shard, FIT_SOURCE[arm])
    named = dict(zip(shard.names, (int(c) for c in allocation.counts)))
    if named != record["class_counts"]:
        raise RuntimeError("Rebuilt training allocation differs from the stored fit")
    fit = fit_multinomial_logistic(
        allocation.x,
        allocation.y,
        lambda_val=extra,
        tol=TOLERANCE,
        max_iter=MAX_ITER,
        sample_weight=allocation.weight,
    )
    scores = {float(c["lambda"]): c["validation_score"] for c in record["candidates"]}
    if select_lambda(scores) != frozen:
        raise RuntimeError("Selection rule does not reproduce the stored choice")
    val_preds, _ = predict_logreg(evals.val_x, fit.coef, fit.intercept)
    scores[extra] = (
        _macro_recall_mean(evals.val_y, val_preds, _cases(evals.val_id))
        if fit.converged
        else None
    )
    chosen = select_lambda(scores)
    test_cases = _cases(evals.test_id)
    frozen_preds = np.asarray(record["splits"]["test"]["preds"])
    refit_preds = predict_logreg(evals.test_x, fit.coef, fit.intercept)[0]
    return {
        "frozen_lambda": frozen,
        "extra_lambda": extra,
        "extra_converged": bool(fit.converged),
        "frozen_validation_score": scores[frozen],
        "extra_validation_score": scores[extra],
        "refit_lambda": chosen,
        "moved": chosen != frozen,
        "test_ba_frozen": 100.0
        * _macro_recall_mean(evals.test_y, frozen_preds, test_cases),
        "test_ba_refit": 100.0
        * _macro_recall_mean(
            evals.test_y, refit_preds if chosen == extra else frozen_preds, test_cases
        ),
    }


def _encoder_split_cells(
    config: dict[str, Any], encoder: str, split_idx: int
) -> list[dict[str, Any]]:
    paths = paths_by_split(config)
    train_df, names, evals, _ = init_shard(config, split_idx, encoder)
    cells = []
    for arm in FIT_SOURCE:
        for s, draw, result_dir in fit_dirs(paths, encoder, arm):
            if s != split_idx:
                continue
            record = require_record(
                result_dir, splits=("test",), array_fields=("preds",)
            )
            if float(record["selected_lambda"]) not in _EXTRA:
                continue
            shard = _shard_context(
                train_df, names, split_idx, draw, patients_per_class(config)
            )
            cell = _refit_cell((shard, evals), record, arm)
            cells.append(
                {"encoder": encoder, "split": s, "draw": draw, "arm": arm, **cell}
            )
    return cells


def run_boundary_refits(config: dict[str, Any]) -> None:
    """Refit every endpoint selection one decade further and write boundary_refits.json."""
    cells = []
    for encoder in ENCODERS:
        # The process-wide feature bank keeps one width; clear it before loading this encoder.
        reset_feature_bank()
        for split_idx in range(N_PATIENT_SPLITS):
            cells.extend(_encoder_split_cells(config, encoder, split_idx))
    write_json(output_root(config) / "data" / "boundary_refits.json", {"cells": cells})
