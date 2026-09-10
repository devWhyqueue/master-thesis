"""One shard = one (split, support) patient-average logistic fit."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from decodability.evidence import CellEvidence, load_cell
from decodability.linear import (
    LinearFitResult,
    fit_multinomial_logistic,
    predict_logreg,
)
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _cluster_discrimination,
)
from imbalance_benchmark.common import ensure_dirs, split_paths, write_run_record

from influence import SUPPORTS, cell_dir, inherited_lambda
from influence.weights import patient_average_weights

__all__ = ["decode_shard_index", "run_fit"]

logger = logging.getLogger(__name__)

FIT_SHARD_COUNT = 3 * len(SUPPORTS)


def decode_shard_index(shard_index: int) -> tuple[int, str]:
    """Decode a fit array index into (split_index, support) over the 3x2 grid."""
    if shard_index not in range(FIT_SHARD_COUNT):
        raise ValueError(f"shard_index must be in [0, {FIT_SHARD_COUNT - 1}]")
    return shard_index // len(SUPPORTS), SUPPORTS[shard_index % len(SUPPORTS)]


def _fit_and_predict(
    config: dict[str, Any], split_index: int, support: str
) -> tuple[CellEvidence, LinearFitResult, np.ndarray, np.ndarray]:
    """Fit the patient-average objective and predict the test split for one cell."""
    lam, _ = inherited_lambda(config, support)
    cell = load_cell(config, split_index, support)
    weights = patient_average_weights(cell.train_y, cell.train_patients)
    fit_res = fit_multinomial_logistic(
        cell.train_x.cpu().numpy(), cell.train_y, lam, sample_weight=weights
    )
    if not fit_res.converged:
        raise RuntimeError(
            f"Patient-average fit did not converge for split {split_index}, "
            f"support {support}"
        )
    preds, probs = predict_logreg(
        cell.test_x.cpu().numpy(), fit_res.coef, fit_res.intercept
    )
    return cell, fit_res, preds, probs


def _build_record(
    config: dict[str, Any],
    param_str: str,
    cell: CellEvidence,
    fit_res: LinearFitResult,
    preds: np.ndarray,
    probs: np.ndarray,
) -> dict[str, Any]:
    """Assemble the run record for one patient-average fit."""
    case_ids = cell.test_identity["case_id"].astype(str).to_numpy()
    slide_ids = cell.test_identity["slide_id"].astype(str).to_numpy()
    endpoints = _cluster_discrimination(
        cell.test_y, preds, case_ids, slide_ids, is_mil=False
    )
    return {
        "dataset": config.get("dataset", {}),
        "feature_extraction": config.get("feature_extraction", {}),
        "method": "logreg",
        "param": param_str,
        "solver": {
            "solver": fit_res.solver,
            "precision": fit_res.precision,
            "tolerance": fit_res.tolerance,
            "solver_tolerance": fit_res.solver_tolerance,
            "max_iter": fit_res.max_iter,
            "n_iter": fit_res.n_iter,
            "objective": fit_res.objective,
            "converged": fit_res.converged,
            "lambda": fit_res.lambda_val,
            "C": fit_res.c_val,
            "weighted": fit_res.weighted,
        },
        "splits": {
            "test": {
                "endpoints": endpoints,
                "labels": cell.test_y,
                "preds": preds,
                "probabilities": probs,
            }
        },
    }


def run_fit(config: dict[str, Any], shard_index: int) -> None:
    """Fit and evaluate the patient-average objective for one (split, support) cell."""
    split_index, support = decode_shard_index(shard_index)
    _, param_str = inherited_lambda(config, support)
    cell, fit_res, preds, probs = _fit_and_predict(config, split_index, support)
    record = _build_record(config, param_str, cell, fit_res, preds, probs)
    paths = split_paths(ensure_dirs(config), split_index)
    write_run_record(cell_dir(paths, support, "patient"), record, keep_arrays=True)
    logger.info(
        "Patient-average fit complete for split %d, support %s", split_index, support
    )
