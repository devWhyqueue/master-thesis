"""B-selected-lambda sensitivity, head/body/tail recall, and lambda/temperature diagnostics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.common import N_PATIENT_SPLITS

from breadth.analyze.secondary import _patient_macro_recalls
from breadth.calibrate import read_temperature

from sites import allocation_dir

from decodability.linear import predict_logreg

from prevalence.fit import class_permutation

from transfer import ARMS, ENCODERS, MAIN_DRAWS
from transfer.analyze.common import fit_dirs, require_record
from transfer.fit import CANDIDATES_NAME, init_shard

__all__ = ["b_lambda_sensitivity", "thirds", "lambda_temperature"]

_SENSITIVITY_ARMS = ("B", "R100", "P100", "S100")


def _sensitivity_preds(
    result_dir: Path, target_lambda: float, test_x: np.ndarray
) -> np.ndarray | None:
    """Predictions from one arm's stored candidate closest to ``target_lambda`` (exact match on
    the shared frozen grid); ``None`` if that candidate never converged."""
    with np.load(result_dir / CANDIDATES_NAME) as data:
        lambdas, coefs, intercepts = data["lambdas"], data["coef"], data["intercept"]
    idx = int(np.argmin(np.abs(lambdas - target_lambda)))
    converged = bool(require_record(result_dir)["candidates"][idx]["converged"])
    if not converged:
        return None
    preds, _ = predict_logreg(test_x, coefs[idx], intercepts[idx])
    return preds


def _encoder_sensitivity(
    config: dict[str, Any],
    paths: dict[int, dict[str, Path]],
    ctxs: dict[int, BootstrapContext],
    n_classes: int,
    encoder: str,
) -> dict[str, Any]:
    arm_ba: dict[str, list[float]] = {a: [] for a in _SENSITIVITY_ARMS}
    skipped = 0
    for s in range(N_PATIENT_SPLITS):
        _, _, evals, _ = init_shard(config, s, encoder)
        for d in MAIN_DRAWS:
            b_lambda = float(
                require_record(allocation_dir(paths[s], f"{encoder}/B", d))[
                    "selected_lambda"
                ]
            )
            for a in _SENSITIVITY_ARMS:
                a_dir = allocation_dir(paths[s], f"{encoder}/{a}", d)
                preds = _sensitivity_preds(a_dir, b_lambda, evals.test_x)
                if preds is None:
                    skipped += 1
                    continue
                recalls = _patient_macro_recalls(
                    ctxs[s], evals.test_y, preds, n_classes
                )
                arm_ba[a].append(float(recalls.mean(axis=0)[0] * 100.0))
    ba_b = float(np.mean(arm_ba["B"])) if arm_ba["B"] else float("nan")
    return {
        "skipped_nonconvergent": skipped,
        **{
            f"D_{a}_b_lambda": (ba_b - float(np.mean(arm_ba[a]))) if arm_ba[a] else None
            for a in ("R100", "P100", "S100")
        },
    }


def b_lambda_sensitivity(
    config: dict[str, Any],
    paths: dict[int, dict[str, Path]],
    ctxs: dict[int, BootstrapContext],
    n_classes: int,
) -> dict[str, Any]:
    """Re-evaluate each encoder's R100/P100/S100 arms at that encoder's own B-selected
    lambda per (split, draw), using stored candidate coefficients (no refit)."""
    return {
        m: _encoder_sensitivity(config, paths, ctxs, n_classes, m) for m in ENCODERS
    }


def thirds(
    paths: dict[int, dict[str, Path]],
    ctxs: dict[int, BootstrapContext],
    names: dict[int, list[str]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean observed patient-macro recall (%) of head/body/tail classes, per encoder/arm."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for m in ENCODERS:
        out[m] = {}
        for arm in ARMS:
            vals: dict[str, list[float]] = {"head": [], "body": [], "tail": []}
            for s, d, result_dir in fit_dirs(paths, m, arm):
                n_classes = len(names[s])
                groups = np.array_split(class_permutation(s, d, n_classes), 3)
                rec = require_record(
                    result_dir, splits=("test",), array_fields=("labels", "preds")
                )
                test = rec["splits"]["test"]
                labels, preds = np.asarray(test["labels"]), np.asarray(test["preds"])
                recalls = (
                    _patient_macro_recalls(ctxs[s], labels, preds, n_classes)[:, 0]
                    * 100.0
                )
                for key, idx in zip(("head", "body", "tail"), groups):
                    vals[key].append(float(recalls[idx].mean()))
            out[m][arm] = {key: float(np.mean(v)) for key, v in vals.items()}
    return out


def lambda_temperature(paths: dict[int, dict[str, Path]]) -> dict[str, Any]:
    """Selected-lambda counts and mean validation temperature per encoder/arm, over every fit."""
    out: dict[str, Any] = {}
    for m in ENCODERS:
        out[m] = {}
        for arm in ARMS:
            lambdas, temps = [], []
            for _, _, result_dir in fit_dirs(paths, m, arm):
                lambdas.append(float(require_record(result_dir)["selected_lambda"]))
                temps.append(read_temperature(result_dir))
            out[m][arm] = {
                "lambda_counts": {str(k): v for k, v in Counter(lambdas).items()},
                "mean_temperature": float(np.mean(temps)),
            }
    return out
