"""Cross-fit and naive lambda-selection oracles: re-prediction only, no refits.

The naive oracle picks lambda and scores it on the same test patients -- winner's curse, worse on
the smaller BRACS test set. The cross-fit oracle avoids this: split test patients into 2 class-
stratified folds, pick lambda on one fold's own rows, predict the other, then swap and concatenate
-- so every prediction comes from a candidate that never saw its own fold's outcome.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from decodability.linear import predict_logreg
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _cluster_discrimination,
    _macro_recall_mean,
    clustered_endpoints,
)
from imbalance_benchmark.common import RUN_RECORD_NAME, write_run_record

from breadth import TIE_TOLERANCE
from breadth.fit import EvalPartition, _build_draw_record

from joint import CENTRE_DEPTH, G
from joint.grid import GridCandidate

__all__ = ["fold_split", "crossfit_predictions", "naive_oracle", "write_oracle_record"]


def fold_split(
    test_id: pd.DataFrame, test_y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified test-patient halves: disjoint row masks covering every test patient."""
    patients = test_id["case_id"].astype(str).to_numpy()
    unique_patients = np.unique(patients)
    patient_class = {
        p: test_y[np.flatnonzero(patients == p)[0]] for p in unique_patients
    }
    rng = np.random.default_rng(seed)
    fold_a: set[str] = set()
    for cls in np.unique(test_y):
        cls_patients = np.array([p for p in unique_patients if patient_class[p] == cls])
        rng.shuffle(cls_patients)
        fold_a.update(cls_patients[: len(cls_patients) // 2].tolist())
    mask_a = np.array([p in fold_a for p in patients])
    return mask_a, ~mask_a


def _pick(
    candidates: list[GridCandidate],
    preds_by_cand: list[np.ndarray],
    test_y: np.ndarray,
    cases: np.ndarray,
    mask: np.ndarray,
) -> int:
    """Best-converged candidate index on ``mask``'s rows, ties toward larger lambda."""
    best_i, best_score = -1, float("-inf")
    for i, cand in enumerate(candidates):
        if not cand.fit.converged:
            continue
        score = _macro_recall_mean(test_y[mask], preds_by_cand[i][mask], cases[mask])
        diff = score - best_score
        if best_i < 0 or diff > TIE_TOLERANCE or abs(diff) <= TIE_TOLERANCE:
            best_i, best_score = i, score
    if best_i < 0:
        raise RuntimeError("No candidate converged")
    return best_i


def _all_predictions(
    candidates: list[GridCandidate], evals: EvalPartition
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    preds, probs = [], []
    for cand in candidates:
        p, pr = predict_logreg(evals.test_x, cand.fit.coef, cand.fit.intercept)
        preds.append(p)
        probs.append(pr)
    return preds, probs


def crossfit_predictions(
    candidates: list[GridCandidate], evals: EvalPartition, seed: int
) -> tuple[np.ndarray, np.ndarray, GridCandidate, GridCandidate]:
    """Full-test (preds, probs) stitched from two folds' own out-of-fold winners."""
    mask_a, mask_b = fold_split(evals.test_id, evals.test_y, seed)
    cases = evals.test_id["case_id"].astype(str).to_numpy()
    preds_by_cand, probs_by_cand = _all_predictions(candidates, evals)
    idx_a = _pick(candidates, preds_by_cand, evals.test_y, cases, mask_a)
    idx_b = _pick(candidates, preds_by_cand, evals.test_y, cases, mask_b)
    test_preds = np.where(mask_a, preds_by_cand[idx_b], preds_by_cand[idx_a])
    test_probs = np.where(mask_a[:, None], probs_by_cand[idx_b], probs_by_cand[idx_a])
    return test_preds, test_probs, candidates[idx_a], candidates[idx_b]


def naive_oracle(
    candidates: list[GridCandidate], evals: EvalPartition
) -> GridCandidate:
    """The single candidate scoring best on the full test set (winner's curse, descriptive only)."""
    cases = evals.test_id["case_id"].astype(str).to_numpy()
    preds_by_cand, _ = _all_predictions(candidates, evals)
    full_mask = np.ones(len(evals.test_y), dtype=bool)
    return candidates[_pick(candidates, preds_by_cand, evals.test_y, cases, full_mask)]


def _val_endpoints(evals: EvalPartition, candidate: GridCandidate) -> dict[str, Any]:
    """Validation endpoints at one candidate's own coefficients (metadata only, not scored)."""
    val_preds, _ = predict_logreg(
        evals.val_x, candidate.fit.coef, candidate.fit.intercept
    )
    v_cases = evals.val_id["case_id"].astype(str).to_numpy()
    v_slides = evals.val_id["slide_id"].astype(str).to_numpy()
    return _cluster_discrimination(
        evals.val_y, val_preds, v_cases, v_slides, is_mil=False
    )


def _oracle_record(
    config: dict[str, Any],
    draw_idx: int,
    winner_a: GridCandidate,
    winner_b: GridCandidate,
    test_preds: np.ndarray,
    test_probs: np.ndarray,
    evals: EvalPartition,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the stitched-prediction run record, metadata keyed to fold A's own winner."""
    val_end = _val_endpoints(evals, winner_a)
    test_end = clustered_endpoints(
        evals.test_y, test_preds, test_probs, evals.test_id, is_mil=False
    )
    rec = _build_draw_record(
        config,
        (G, CENTRE_DEPTH, draw_idx),
        winner_a.lambda_val,
        winner_a.fit,
        (test_preds, test_probs, val_end, test_end),
        evals.test_y,
    )
    return {
        **rec,
        **extra,
        "fold_lambda_a": winner_a.lambda_val,
        "fold_lambda_b": winner_b.lambda_val,
    }


def write_oracle_record(
    config: dict[str, Any],
    out_dir: Path,
    candidates: list[GridCandidate],
    evals: EvalPartition,
    draw_idx: int,
    seed: int,
    extra: dict[str, Any],
) -> None:
    """Cross-fit oracle's stitched test predictions, written as a run record (no refit)."""
    if (out_dir / RUN_RECORD_NAME).exists():
        return
    test_preds, test_probs, winner_a, winner_b = crossfit_predictions(
        candidates, evals, seed
    )
    rec = _oracle_record(
        config, draw_idx, winner_a, winner_b, test_preds, test_probs, evals, extra
    )
    write_run_record(out_dir, rec, keep_arrays=True)
