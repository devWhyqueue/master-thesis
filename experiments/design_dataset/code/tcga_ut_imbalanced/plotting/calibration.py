import json
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from tcga_ut_imbalanced.plotting import (
    _benchmark,
    _mean_std,
    _tex,
    _write_table,
    _write_unavailable,
)

CALIBRATION_METRICS = (
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
)


def calibration_summary(results_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Return raw and temperature-scaled calibration metrics per run."""
    selection_path = output_dir / "tuning_selection.json"
    if not selection_path.exists():
        return pd.DataFrame()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection:
        return pd.DataFrame()
    return pd.DataFrame(_calibration_runs(results_dir, selection))


def _calibration_runs(results_dir: Path, selection: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for entry in selection:
        rows.extend(_entry_calibration_rows(entry, results_dir))
    return rows


def _entry_calibration_rows(entry: dict, results_dir: Path) -> list[dict]:
    m = re.match(r"order=(?P<order>.+)/param=(?P<parameter>[\d.]+)", entry["regime"])
    if m is None:
        return []
    base = {
        "method": entry["method"],
        "order": m.group("order"),
        "parameter": m.group("parameter"),
    }
    rows = []
    for seed in range(3):
        run_dir = (
            results_dir
            / "tuning"
            / entry["benchmark"]
            / entry["regime"]
            / entry["method"]
            / entry["variant"]
            / f"seed={seed}"
        )
        val = run_dir / "validation_results.json"
        test = run_dir / "test_results.json"
        if not val.exists() or not test.exists():
            continue
        row = _calibration_row({**base, "seed": str(seed)}, val, test)
        if row:
            rows.append(row)
    return rows


def write_calibration_tables(frame: pd.DataFrame, tables_dir: Path) -> None:
    """Write calibration tables for patch and WSI-bag benchmarks."""
    for benchmark, filename in [
        ("patch", "result_calibration_patch.tex"),
        ("wsi_bag", "result_calibration_wsi_bag.tex"),
    ]:
        part = (
            cast(pd.DataFrame, frame[frame["benchmark"] == benchmark])
            if not frame.empty
            else frame
        )
        write_calibration_table(part, tables_dir / filename)


def write_calibration_table(frame: pd.DataFrame, path: Path) -> None:
    """Write one benchmark calibration table."""
    header = "Method & NLL & NLL+TS & Brier & Brier+TS & ECE & ECE+TS & $T$"
    if frame.empty:
        _write_unavailable(path, header)
        return
    aggregate = _aggregate_calibration(frame)
    rows = [_calibration_table_row(row) for _, row in aggregate.iterrows()]
    _write_table(path, header, rows)


def _aggregate_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    raw_agg = {f"{m}_{s}": (m, s) for m in _raw_metrics() for s in ("mean", "std")}
    scaled_agg = {
        f"{m}_scaled_{s}": (f"{m}_scaled", s)
        for m in CALIBRATION_METRICS
        for s in ("mean", "std")
    }
    grouped = frame.groupby(["method", "order", "parameter"]).agg(
        **raw_agg, **scaled_agg
    )
    return cast(pd.DataFrame, grouped.reset_index())


def _calibration_table_row(row: pd.Series) -> str:
    return (
        f"{_tex(row['method'])} ({_tex(row['order'])}, "
        f"$\\lambda={float(row['parameter']):.1f}$) & "
        f"{_mean_std(row, 'negative_log_likelihood')} & "
        f"{_mean_std(row, 'negative_log_likelihood_scaled')} & "
        f"{_mean_std(row, 'brier_score')} & "
        f"{_mean_std(row, 'brier_score_scaled')} & "
        f"{_mean_std(row, 'expected_calibration_error')} & "
        f"{_mean_std(row, 'expected_calibration_error_scaled')} & "
        f"{_mean_std(row, 'temperature')}\\\\"
    )


def _raw_metrics() -> tuple[str, ...]:
    return (*CALIBRATION_METRICS, "temperature")


def _calibration_row(
    metadata: dict[str, str], validation_path: Path, test_path: Path
) -> dict[str, object]:
    validation = _load_result_payload(validation_path)
    test = _load_result_payload(test_path)
    if not validation or not test:
        return {}
    temperature = _fit_temperature(
        cast(np.ndarray, validation["labels"]),
        cast(np.ndarray, validation["probabilities"]),
    )
    return _calibration_result(metadata, test, temperature)


def _calibration_result(
    metadata: dict[str, str], test: dict[str, np.ndarray], temperature: float
) -> dict[str, object]:
    labels = cast(np.ndarray, test["labels"])
    probabilities = cast(np.ndarray, test["probabilities"])
    scaled = _temperature_scale(probabilities, temperature)
    method = metadata["method"]
    return {
        "method": method,
        "benchmark": _benchmark(method),
        "order": metadata["order"],
        "parameter": float(metadata["parameter"]),
        "seed": int(metadata["seed"]),
        "temperature": temperature,
        **_calibration_metrics(labels, probabilities),
        **{
            f"{key}_scaled": value
            for key, value in _calibration_metrics(labels, scaled).items()
        },
    }


def _load_result_payload(path: Path) -> dict[str, np.ndarray]:
    with open(path) as file:
        payload = json.load(file)
    if "labels" not in payload or "probabilities" not in payload:
        return {}
    probabilities = np.asarray(payload["probabilities"], dtype="float64")
    labels = np.asarray(payload["labels"], dtype="int64")
    if probabilities.ndim != 2 or labels.ndim != 1 or len(labels) == 0:
        return {}
    return {"labels": labels, "probabilities": probabilities}


def _fit_temperature(labels: np.ndarray, probabilities: np.ndarray) -> float:
    log_t = np.linspace(np.log(0.05), np.log(10.0), 160)
    losses = [
        _negative_log_likelihood(
            labels, _temperature_scale(probabilities, float(np.exp(v)))
        )
        for v in log_t
    ]
    return float(np.exp(log_t[int(np.argmin(losses))]))


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    logits = np.log(clipped) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _calibration_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    return {
        "negative_log_likelihood": _negative_log_likelihood(labels, probabilities),
        "brier_score": _brier_score(labels, probabilities),
        "expected_calibration_error": _expected_calibration_error(
            labels, probabilities
        ),
    }


def _negative_log_likelihood(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(labels)), labels])))


def _brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return sum(
        _bin_error(confidences, correct, lower, upper)
        for lower, upper in zip(edges[:-1], edges[1:])
    )


def _bin_error(
    confidences: np.ndarray, correct: np.ndarray, lower: float, upper: float
) -> float:
    mask = (confidences > lower) & (confidences <= upper)
    if not bool(mask.any()):
        return 0.0
    return float(mask.mean()) * abs(
        float(correct[mask].mean()) - float(confidences[mask].mean())
    )
