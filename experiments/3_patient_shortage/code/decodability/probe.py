"""Probe drivers for validation candidate evaluation and locked test evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    clustered_endpoints,
)
from imbalance_benchmark.common import (
    ensure_dirs,
    split_paths,
    verify_signed_file,
    write_run_record,
)

from decodability import K_VALUES, LAMBDAS, probe_dir
from decodability.evidence import load_cell
from decodability.linear import (
    LinearFitResult,
    fit_multinomial_logistic,
    predict_logreg,
)
from decodability.neighbours import top_neighbours, vote

__all__ = ["run_probe_val", "run_probe_test"]

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
    safe_probs = np.clip(probs, 1e-12, 1.0)
    safe_probs = safe_probs / np.sum(safe_probs, axis=1, keepdims=True)
    endpoints = clustered_endpoints(
        labels=labels, predictions=preds, probabilities=safe_probs, identity=identity
    )
    discrim = {
        k: v
        for k, v in endpoints.items()
        if "nll" not in k and "brier" not in k and "calibration" not in k
    }
    record = {
        **meta,
        **(extra_fields or {}),
        "splits": {
            split_name: {
                "endpoints": discrim,
                "labels": labels.tolist(),
                "preds": preds.tolist(),
                "probabilities": probs.tolist(),
            }
        },
    }
    write_run_record(result_dir, record, keep_arrays=True)


def _val_logreg(
    cell: Any, paths: dict[str, Path], config: dict[str, Any], support: str
) -> None:
    train_x, val_x = cell.train_x.cpu().numpy(), cell.val_x.cpu().numpy()
    coefs, intercepts = {}, {}
    for lam in LAMBDAS:
        fit_res: LinearFitResult = fit_multinomial_logistic(train_x, cell.train_y, lam)
        coefs[str(lam)], intercepts[str(lam)] = fit_res.coef, fit_res.intercept
        val_preds, val_probs = predict_logreg(val_x, fit_res.coef, fit_res.intercept)
        r_dir = probe_dir(paths, support, "logreg", f"lambda={lam}")
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
            r_dir,
            _record_meta(config, "logreg", f"lambda={lam}"),
            "validation",
            cell.val_y,
            val_preds,
            val_probs,
            cell.val_identity,
            extra,
        )
    coef_file = paths["data"] / f"logreg_coefficients_{support}.npz"
    coef_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        coef_file,
        **{f"coef_{k}": v for k, v in coefs.items()},
        **{f"intercept_{k}": v for k, v in intercepts.items()},
    )


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


def run_probe_val(config: dict[str, Any], split_index: int, support: str) -> None:
    """Fit validation candidates and evaluate for both probes."""
    paths = split_paths(ensure_dirs(config), split_index)
    cell = load_cell(config, split_index, support)
    _val_logreg(cell, paths, config, support)
    _val_knn(cell, paths, config, support)
    logger.info(
        "Probe validation complete for split %d, support %s", split_index, support
    )


def _test_logreg(
    cell: Any,
    paths: dict[str, Path],
    config: dict[str, Any],
    support: str,
    selected: dict[str, Any],
) -> None:
    lam_float, param_str = float(selected["selected"]), selected["selected_param_str"]
    coef_file = paths["data"] / f"logreg_coefficients_{support}.npz"
    with np.load(coef_file) as data:
        coef, intercept = data[f"coef_{lam_float}"], data[f"intercept_{lam_float}"]
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
