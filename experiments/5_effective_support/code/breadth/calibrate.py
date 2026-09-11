"""Post-hoc temperature scaling of each selected fit, fitted on validation data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from decodability.linear import fit_multinomial_logistic
from imbalance_benchmark.analysis.calibration import apply_temperature, fit_temperature
from imbalance_benchmark.analysis.query import read_run_record
from imbalance_benchmark.common import write_json

from breadth import MAX_ITER, N_DRAWS, TOLERANCE, draw_dir
from breadth.fit import (
    EvalPartition,
    decode_shard_index,
    draw_training_data,
    init_shard,
)

__all__ = [
    "TEMPERATURE_NAME",
    "read_temperature",
    "run_calibrate_shard",
    "scale_stored_probabilities",
    "scaled_test_probabilities",
]

logger = logging.getLogger(__name__)

TEMPERATURE_NAME = "temperature.json"
SCALED_NAME = "temperature_scaled.npz"
# The refit repeats a deterministic solve; only floating-point ties may flip.
PRED_AGREEMENT_FLOOR = 0.999
RECONSTRUCTION_TOLERANCE = 1e-6


def scale_stored_probabilities(
    probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    """Temperature-scale stored float64 softmax outputs.

    Their logarithm equals the logits up to a per-row constant, which the
    softmax removes, so no logits need to be stored.
    """
    tiny = np.finfo(np.float64).tiny
    return apply_temperature(np.log(np.maximum(probabilities, tiny)), temperature)


def read_temperature(result_dir: Path) -> float:
    """Read the validation-fitted temperature of one draw."""
    payload = json.loads((result_dir / TEMPERATURE_NAME).read_text(encoding="utf-8"))
    return float(payload["temperature"])


def _logits(
    features: np.ndarray, coef: np.ndarray, intercept: np.ndarray
) -> np.ndarray:
    """Linear multinomial logits in float64."""
    return np.asarray(features, dtype=np.float64) @ coef.T + intercept


def scaled_test_probabilities(
    result_dir: Path, probabilities: np.ndarray
) -> np.ndarray:
    """Temperature-scaled test probabilities of one draw, exact where stored."""
    exact = result_dir / SCALED_NAME
    if exact.exists():
        with np.load(exact) as arrays:
            return arrays["probabilities"]
    return scale_stored_probabilities(probabilities, read_temperature(result_dir))


def _calibration_record(
    evals: EvalPartition,
    val_logits: np.ndarray,
    test_logits: np.ndarray,
    stored_preds: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Fit the temperature after verifying the refit; return exact scaled test output."""
    agreement = float(np.mean(test_logits.argmax(axis=1) == stored_preds))
    if agreement < PRED_AGREEMENT_FLOOR:
        raise RuntimeError(f"Refit reproduces only {agreement:.4%} of test predictions")
    temperature = fit_temperature(val_logits, evals.val_y).temperature
    record = {"temperature": temperature, "test_pred_agreement": agreement}
    return record, apply_temperature(test_logits, temperature)


def _store_if_inexact(
    result_dir: Path, scaled: np.ndarray, stored: np.ndarray, temperature: float
) -> float:
    """Keep exact scaled output where underflowed stored probabilities cannot rescale."""
    error = float(
        np.max(np.abs(scaled - scale_stored_probabilities(stored, temperature)))
    )
    if error > RECONSTRUCTION_TOLERANCE:
        np.savez_compressed(result_dir / SCALED_NAME, probabilities=scaled)
    return error


def _calibrate_draw(
    shard: tuple[pd.DataFrame, list[str], EvalPartition, dict[str, Path]],
    meta: tuple[int, int, int, int],
) -> None:
    """Refit one draw at its selected lambda and store its validation temperature."""
    train_df, classes, evals, paths = shard
    _, g, m, draw_idx = meta
    result_dir = draw_dir(paths, g, m, draw_idx)
    record = read_run_record(
        result_dir, splits=("test",), array_fields=("preds", "probabilities")
    )
    if record is None:
        raise RuntimeError(f"Missing run record at {result_dir}")
    lam = float(record["selected_lambda"])
    train_x, train_y = draw_training_data(train_df, classes, meta)
    fit = fit_multinomial_logistic(
        train_x, train_y, lambda_val=lam, tol=TOLERANCE, max_iter=MAX_ITER
    )
    test = record["splits"]["test"]
    payload, scaled = _calibration_record(
        evals,
        _logits(evals.val_x, fit.coef, fit.intercept),
        _logits(evals.test_x, fit.coef, fit.intercept),
        np.asarray(test["preds"]),
    )
    payload["max_abs_scaled_probability_error"] = _store_if_inexact(
        result_dir, scaled, np.asarray(test["probabilities"]), payload["temperature"]
    )
    write_json(result_dir / TEMPERATURE_NAME, {**payload, "selected_lambda": lam})


def run_calibrate_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit a validation temperature for every draw of one (split, G, m) shard."""
    split_idx, g, m = decode_shard_index(shard_index)
    shard = init_shard(config, split_idx)
    for draw_idx in range(N_DRAWS):
        logger.info(
            "Calibrating split %d, G=%d, m=%d, draw %d", split_idx, g, m, draw_idx
        )
        _calibrate_draw(shard, (split_idx, g, m, draw_idx))
