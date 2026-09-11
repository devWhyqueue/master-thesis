"""Model fitting and regularization tuning for one (split, cell) shard."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from decodability.evidence import load_freeze_meta
from decodability.linear import (
    LinearFitResult,
    fit_multinomial_logistic,
    predict_logreg,
)
from imbalance_benchmark.analysis.query import load_test_identity
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _cluster_discrimination,
    clustered_endpoints,
)
from imbalance_benchmark.common import (
    ensure_dirs,
    split_paths,
    write_run_record,
)
from imbalance_benchmark.datasets.data import ImbalanceDataset
from imbalance_benchmark.datasets.features import load_feature_row

from breadth import (
    FIT_SHARD_COUNT,
    GRID_CELLS,
    LAMBDAS,
    MAX_ITER,
    N_DRAWS,
    TIE_TOLERANCE,
    TOLERANCE,
    draw_dir,
    exp2_split_paths,
)
from breadth.sampling import sample_cell_draw

__all__ = ["EvalPartition", "decode_shard_index", "run_fit_shard", "tune_and_fit_draw"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalPartition:
    """Validation and test partitions and patient identities."""

    val_x: np.ndarray
    val_y: np.ndarray
    val_id: pd.DataFrame
    test_x: np.ndarray
    test_y: np.ndarray
    test_id: pd.DataFrame


def decode_shard_index(shard_index: int) -> tuple[int, int, int]:
    """Decode a shard index into (split_index, g, m) over the 3x3x3 grid."""
    if shard_index not in range(FIT_SHARD_COUNT):
        raise ValueError(f"shard_index must be in [0, {FIT_SHARD_COUNT - 1}]")
    s_idx = shard_index // len(GRID_CELLS)
    c_idx = shard_index % len(GRID_CELLS)
    g, m = GRID_CELLS[c_idx]
    return s_idx, g, m


def _load_features_for_df(df: pd.DataFrame) -> np.ndarray:
    """Load features for a slice of manifest rows directly."""
    paths = df["feature_path"].astype(str).to_numpy()
    f_idx = df["feature_index"].to_numpy() if "feature_index" in df.columns else None
    rows = [
        load_feature_row(
            paths[i],
            int(f_idx[i]) if f_idx is not None and pd.notna(f_idx[i]) else None,
        )
        for i in range(len(df))
    ]
    return torch.stack(rows).numpy()


def _load_eval_partition(
    manifest_path: str | Path, class_names: list[str], split_name: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load features, integer targets, and patient identity for validation/test."""
    ds = ImbalanceDataset(manifest_path, split_name=split_name, class_names=class_names)
    return (
        _load_features_for_df(ds.df),
        ds.get_int_targets(),
        load_test_identity(manifest_path, is_mil=False, split_name=split_name),
    )


def _select_best_lambda(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_id: pd.DataFrame,
) -> tuple[LinearFitResult, float, dict[str, Any]]:
    """Grid-search lambda on validation, breaking ties toward larger lambda."""
    v_cases = val_id["case_id"].astype(str).to_numpy()
    v_slides = val_id["slide_id"].astype(str).to_numpy()
    best_score, best_lam = -1.0, LAMBDAS[0]
    best_fit: LinearFitResult | None = None
    best_end: dict[str, Any] = {}

    for lam in LAMBDAS:
        fit = fit_multinomial_logistic(
            train_x, train_y, lambda_val=lam, tol=TOLERANCE, max_iter=MAX_ITER
        )
        if not fit.converged:
            continue
        preds, _ = predict_logreg(val_x, fit.coef, fit.intercept)
        end = _cluster_discrimination(val_y, preds, v_cases, v_slides, is_mil=False)
        score = float(end["patient_macro_balanced_accuracy"])
        diff = score - best_score
        if best_fit is None or diff > TIE_TOLERANCE or abs(diff) <= TIE_TOLERANCE:
            best_score, best_lam, best_fit, best_end = score, lam, fit, end

    if best_fit is None:
        raise RuntimeError("No candidate converged during validation tuning")
    return best_fit, best_lam, best_end


def tune_and_fit_draw(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evals: EvalPartition,
) -> tuple[
    LinearFitResult, float, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]
]:
    """Select best lambda on validation and evaluate on test."""
    best_fit, best_lam, val_end = _select_best_lambda(
        train_x, train_y, evals.val_x, evals.val_y, evals.val_id
    )
    test_preds, test_probs = predict_logreg(
        evals.test_x, best_fit.coef, best_fit.intercept
    )
    # Full endpoint set, so a run record carries the prespecified probability-quality
    # endpoints without a later pass over the stored arrays.
    test_end = clustered_endpoints(
        evals.test_y, test_preds, test_probs, evals.test_id, is_mil=False
    )
    return best_fit, best_lam, test_preds, test_probs, val_end, test_end


def _build_draw_record(
    config: dict[str, Any],
    grid_meta: tuple[int, int, int],
    best_lam: float,
    fit: LinearFitResult,
    eval_outs: tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]],
    test_y: np.ndarray,
) -> dict[str, Any]:
    """Assemble the run record dict for one draw."""
    g, m, draw_idx = grid_meta
    test_preds, test_probs, val_end, test_end = eval_outs
    return {
        "dataset": config.get("dataset", {}),
        "feature_extraction": config.get("feature_extraction", {}),
        "method": "logreg",
        "param": f"lambda={best_lam}",
        "grid": {"g": g, "m": m, "draw": draw_idx},
        "selected_lambda": best_lam,
        "solver": {
            "solver": fit.solver,
            "precision": fit.precision,
            "tolerance": fit.tolerance,
            "solver_tolerance": fit.solver_tolerance,
            "max_iter": fit.max_iter,
            "n_iter": fit.n_iter,
            "objective": fit.objective,
            "converged": fit.converged,
            "lambda": fit.lambda_val,
            "C": fit.c_val,
        },
        "splits": {
            "validation": {"endpoints": val_end},
            "test": {
                "endpoints": test_end,
                "labels": test_y,
                "preds": test_preds,
                "probabilities": test_probs,
            },
        },
    }


def _run_one_draw(
    config: dict[str, Any],
    paths: dict[str, Path],
    train_df: pd.DataFrame,
    class_names: list[str],
    meta: tuple[int, int, int, int],
    evals: EvalPartition,
) -> None:
    """Execute validation selection and test evaluation for one draw."""
    split_idx, g, m, draw_idx = meta
    sample_df = sample_cell_draw(
        train_df, class_names, g, m, split_idx, draw_idx, base_seed=0
    )
    cmap = {name: i for i, name in enumerate(class_names)}
    train_y = np.array([cmap[c] for c in sample_df["cancer_type"]], dtype=np.int64)
    train_x = _load_features_for_df(sample_df)

    fit, lam, t_p, t_pr, v_e, t_e = tune_and_fit_draw(train_x, train_y, evals)
    rec = _build_draw_record(
        config, (g, m, draw_idx), lam, fit, (t_p, t_pr, v_e, t_e), evals.test_y
    )
    write_run_record(draw_dir(paths, g, m, draw_idx), rec, keep_arrays=True)


def _init_shard(
    config: dict[str, Any], split_idx: int
) -> tuple[pd.DataFrame, list[str], EvalPartition, dict[str, Path]]:
    """Load train manifest, class names, and validation/test partitions."""
    exp2_p = exp2_split_paths(config, split_idx)
    m_file = exp2_p["data"] / "manifest.csv"
    classes = list(load_freeze_meta(exp2_p)["class_names"])
    train_df = pd.read_csv(m_file).query("split == 'train'").reset_index(drop=True)
    val_data = _load_eval_partition(m_file, classes, "validation")
    test_data = _load_eval_partition(m_file, classes, "test")
    evals = EvalPartition(*val_data, *test_data)
    paths = split_paths(ensure_dirs(config), split_idx)
    return train_df, classes, evals, paths


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Execute all draws for one (split, G, m) shard."""
    split_idx, g, m = decode_shard_index(shard_index)
    train_df, classes, evals, paths = _init_shard(config, split_idx)
    for draw_idx in range(N_DRAWS):
        logger.info("Fitting split %d, G=%d, m=%d, draw %d", split_idx, g, m, draw_idx)
        _run_one_draw(
            config, paths, train_df, classes, (split_idx, g, m, draw_idx), evals
        )
