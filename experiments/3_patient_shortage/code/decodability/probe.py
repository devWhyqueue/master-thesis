"""Probe drivers for validation candidate evaluation and locked test evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _cluster_discrimination,
)
from imbalance_benchmark.common import (
    ensure_dirs,
    split_paths,
    verify_signed_file,
    write_run_record,
)

from decodability import K_VALUES, LAMBDAS, SUPPORTS, probe_dir
from decodability.evidence import load_cell
from decodability.linear import (
    LinearFitResult,
    fit_multinomial_logistic,
    predict_logreg,
)
from decodability.neighbours import top_neighbours, vote

__all__ = ["decode_shard_index", "run_probe_val", "run_probe_test"]

logger = logging.getLogger(__name__)


def _record_meta(config: dict[str, Any], readout: str, param: str) -> dict[str, Any]:
    return {
        "dataset": config.get("dataset", {}),
        "feature_extraction": config.get("feature_extraction", {}),
        "method": readout,
        "param": param,
    }


def _save_split_record(
    result_dir: Path,
    meta: dict[str, Any],
    split_name: str,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    identity: pd.DataFrame,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    case_ids = identity["case_id"].astype(str).to_numpy()
    slide_ids = identity["slide_id"].astype(str).to_numpy()
    discrim = _cluster_discrimination(labels, preds, case_ids, slide_ids, is_mil=False)
    record = {
        **meta,
        **(extra_fields or {}),
        "splits": {
            split_name: {
                "endpoints": discrim,
                "labels": labels,
                "preds": preds,
                "probabilities": probs,
            }
        },
    }
    write_run_record(result_dir, record, keep_arrays=True)


def _val_logreg(
    cell: Any, paths: dict[str, Path], config: dict[str, Any], support: str, lam: float
) -> None:
    train_x, val_x = cell.train_x.cpu().numpy(), cell.val_x.cpu().numpy()
    param_str = f"lambda={lam}"
    fit_res: LinearFitResult = fit_multinomial_logistic(train_x, cell.train_y, lam)
    val_preds, val_probs = predict_logreg(val_x, fit_res.coef, fit_res.intercept)
    extra = {
        "solver": {
            "solver": fit_res.solver,
            "precision": fit_res.precision,
            "tolerance": fit_res.tolerance,
            "max_iter": fit_res.max_iter,
            "n_iter": fit_res.n_iter,
            "objective": fit_res.objective,
            "converged": fit_res.converged,
            "lambda": fit_res.lambda_val,
            "C": fit_res.c_val,
        }
    }
    _save_split_record(
        probe_dir(paths, support, "logreg", param_str),
        _record_meta(config, "logreg", param_str),
        "validation",
        cell.val_y,
        val_preds,
        val_probs,
        cell.val_identity,
        extra,
    )
    coef_file = paths["data"] / f"logreg_coefficients_{support}_{param_str}.npz"
    coef_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(coef_file, coef=fit_res.coef, intercept=fit_res.intercept)


def _val_knn(
    cell: Any, paths: dict[str, Path], config: dict[str, Any], support: str
) -> None:
    order = np.argsort(cell.train_patches)
    indices, _, stats = top_neighbours(cell.val_x, cell.train_x[order])
    val_labels, n_cls = cell.train_y[order][indices], len(cell.class_names)
    for k in K_VALUES:
        preds, probs = vote(val_labels, k, n_cls)
        extra = {
            "search_stats": {
                "wall_time_s": stats.wall_time_s,
                "peak_memory_mb": stats.peak_memory_mb,
                "device": stats.device,
                "query_batch_size": stats.query_batch_size,
                "bank_batch_size": stats.bank_batch_size,
            }
        }
        _save_split_record(
            probe_dir(paths, support, "knn", f"k={k}"),
            _record_meta(config, "knn", f"k={k}"),
            "validation",
            cell.val_y,
            preds,
            probs,
            cell.val_identity,
            extra,
        )


def decode_shard_index(shard_index: int) -> tuple[int, str, int]:
    """Decode a probe-val array index into (split_index, support, unit).

    Layout: ``split = idx // 16``, ``support = SUPPORTS[(idx % 16) // 8]``,
    ``unit = idx % 8`` where units 0..6 index ``LAMBDAS`` and unit 7 runs the
    k-NN search. One task per lambda keeps each SLURM task's wall clock
    bounded by a single L-BFGS fit instead of the whole lambda grid.
    """
    split_index = shard_index // 16
    support = SUPPORTS[(shard_index % 16) // 8]
    unit = shard_index % 8
    return split_index, support, unit


def run_probe_val(config: dict[str, Any], shard_index: int) -> None:
    """Fit one validation candidate (one lambda, or the k-NN search) for one shard."""
    split_index, support, unit = decode_shard_index(shard_index)
    paths = split_paths(ensure_dirs(config), split_index)
    cell = load_cell(config, split_index, support)
    if unit < len(LAMBDAS):
        _val_logreg(cell, paths, config, support, LAMBDAS[unit])
    else:
        _val_knn(cell, paths, config, support)
    logger.info(
        "Probe validation complete for split %d, support %s, unit %d",
        split_index,
        support,
        unit,
    )


def _test_logreg(
    cell: Any,
    paths: dict[str, Path],
    config: dict[str, Any],
    support: str,
    selected: dict[str, Any],
) -> None:
    param_str = selected["selected_param_str"]
    coef_file = paths["data"] / f"logreg_coefficients_{support}_{param_str}.npz"
    with np.load(coef_file) as data:
        coef, intercept = data["coef"], data["intercept"]
    test_preds, test_probs = predict_logreg(cell.test_x.cpu().numpy(), coef, intercept)
    val_r_dir = probe_dir(paths, support, "logreg", param_str)
    _save_split_record(
        val_r_dir,
        _record_meta(config, "logreg", param_str),
        "test",
        cell.test_y,
        test_preds,
        test_probs,
        cell.test_identity,
    )


def _test_knn(
    cell: Any,
    paths: dict[str, Path],
    config: dict[str, Any],
    support: str,
    selected: dict[str, Any],
) -> None:
    order = np.argsort(cell.train_patches)
    k_int, param_str = int(selected["selected"]), selected["selected_param_str"]
    indices, _, stats = top_neighbours(cell.test_x, cell.train_x[order], top=k_int)
    preds, probs = vote(cell.train_y[order][indices], k_int, len(cell.class_names))
    extra = {
        "search_stats": {
            "wall_time_s": stats.wall_time_s,
            "peak_memory_mb": stats.peak_memory_mb,
            "device": stats.device,
            "query_batch_size": stats.query_batch_size,
            "bank_batch_size": stats.bank_batch_size,
        }
    }
    _save_split_record(
        probe_dir(paths, support, "knn", param_str),
        _record_meta(config, "knn", param_str),
        "test",
        cell.test_y,
        preds,
        probs,
        cell.test_identity,
        extra,
    )


def run_probe_test(config: dict[str, Any], split_index: int, support: str) -> None:
    """Evaluate locked hyperparameters on test split."""
    root_p = ensure_dirs(config)["root"]
    sel_path = root_p / "data" / "probe_selection.json"
    verify_signed_file(sel_path)
    supp_sel = (
        json.loads(sel_path.read_text(encoding="utf-8"))
        .get("supports", {})
        .get(support, {})
    )
    paths = split_paths(ensure_dirs(config), split_index)
    cell = load_cell(config, split_index, support)
    _test_logreg(cell, paths, config, support, supp_sel["logreg"])
    _test_knn(cell, paths, config, support, supp_sel["knn"])
    logger.info("Probe test complete for split %d, support %s", split_index, support)
