import json
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from analysis.plotting import (
    PATCH_ORDER,
    WSI_ORDER,
    _benchmark,
    _mean_std,
    _method_label,
    _write_table,
    _write_unavailable,
    _write_wide_table,
    brier_score,
    calibration_headline_colspec_headers,
    calibration_t_cell,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    temperature_scale,
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
        val, test = run_dir / "validation_results.json", run_dir / "test_results.json"
        if not val.exists() or not test.exists():
            continue
        row = _calibration_row({**base, "seed": str(seed)}, val, test)
        if row:
            rows.append(row)
    return rows


def _aggregate_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate calibration metrics across seeds."""
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


def _raw_metrics() -> tuple[str, ...]:
    return (*CALIBRATION_METRICS, "temperature")


def _calibration_row(
    metadata: dict[str, str], validation_path: Path, test_path: Path
) -> dict[str, object]:
    validation = _load_result_payload(validation_path)
    test = _load_result_payload(test_path)
    if not validation or not test:
        return {}
    temp = fit_temperature(
        cast(np.ndarray, validation["labels"]),
        cast(np.ndarray, validation["probabilities"]),
    )
    return _calibration_result(metadata, test, temp)


def _calibration_result(
    metadata: dict[str, str], test: dict[str, np.ndarray], temp: float
) -> dict[str, object]:
    labels = cast(np.ndarray, test["labels"])
    probs = cast(np.ndarray, test["probabilities"])
    scaled = temperature_scale(probs, temp)
    method = metadata["method"]
    return {
        "method": method,
        "benchmark": _benchmark(method),
        "order": metadata["order"],
        "parameter": float(metadata["parameter"]),
        "seed": int(metadata["seed"]),
        "temperature": temp,
        **_calibration_metrics(labels, probs),
        **{f"{k}_scaled": v for k, v in _calibration_metrics(labels, scaled).items()},
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


def _calibration_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    return {
        "negative_log_likelihood": negative_log_likelihood(labels, probabilities),
        "brier_score": brier_score(labels, probabilities),
        "expected_calibration_error": expected_calibration_error(labels, probabilities),
    }


def write_calibration_tables(frame: pd.DataFrame, tables_dir: Path) -> None:
    """Write headline ECE+T and full calibration tables for patch and WSI-bag benchmarks."""
    for benchmark, stem in [
        ("patch", "result_calibration_patch"),
        ("wsi_bag", "result_calibration_wsi_bag"),
    ]:
        part = (
            cast(pd.DataFrame, frame[frame["benchmark"] == benchmark])
            if not frame.empty
            else frame
        )
        write_calibration_table(part, tables_dir / f"{stem}.tex", benchmark == "patch")
        write_calibration_full_table(part, tables_dir / f"{stem}_full.tex")


def write_calibration_table(frame: pd.DataFrame, path: Path, is_patch: bool) -> None:
    """Write headline calibration table: ECE (raw and TS) and fitted T, lambda as columns."""
    params = [0.8, 1.1, 1.3]
    fallback = (
        "Method & "
        + " & ".join(f"ECE & ECE+TS ($\\lambda={p:.1f}$)" for p in params)
        + " & $T$"
    )
    if frame.empty:
        _write_unavailable(path, fallback)
        return
    aggregate = _aggregate_calibration(frame)
    method_order = PATCH_ORDER if is_patch else WSI_ORDER
    methods = list(aggregate["method"].unique())
    ordered = [m for m in method_order if m in methods] + [
        m for m in methods if m not in method_order
    ]
    colspec, header_lines = calibration_headline_colspec_headers(params)
    rows = [
        _calibration_headline_row(
            m, cast(pd.DataFrame, aggregate[aggregate["method"] == m]), params
        )
        for m in ordered
    ]
    _write_wide_table(path, colspec, header_lines, rows)


def write_calibration_full_table(frame: pd.DataFrame, path: Path) -> None:
    """Write full 7-metric calibration table for appendix (one row per method x lambda)."""
    header = (
        "Method & $\\lambda$ & NLL & NLL+TS & Brier & Brier+TS & ECE & ECE+TS & $T$"
    )
    if frame.empty:
        _write_unavailable(path, header)
        return
    aggregate = _aggregate_calibration(frame)
    rows = [_calibration_full_row(row) for _, row in aggregate.iterrows()]
    _write_table(path, header, rows)


def _calibration_headline_row(
    method: str, part: pd.DataFrame, params: list[float]
) -> str:
    cells = [_method_label(method)]
    t_vals = []
    for p in params:
        p_row = part[part["parameter"] == p]
        if p_row.empty:
            cells += ["--", "--"]
            t_vals.append(None)
        else:
            r = p_row.iloc[0]
            cells += [
                _mean_std(r, "expected_calibration_error"),
                _mean_std(r, "expected_calibration_error_scaled"),
            ]
            t_vals.append(r)
    cells.append(calibration_t_cell(t_vals))
    return " & ".join(cells) + "\\\\"


def _calibration_full_row(row: pd.Series) -> str:
    return (
        f"{_method_label(str(row['method']))} & $\\lambda={float(row['parameter']):.1f}$ & "
        f"{_mean_std(row, 'negative_log_likelihood')} & {_mean_std(row, 'negative_log_likelihood_scaled')} & "
        f"{_mean_std(row, 'brier_score')} & {_mean_std(row, 'brier_score_scaled')} & "
        f"{_mean_std(row, 'expected_calibration_error')} & {_mean_std(row, 'expected_calibration_error_scaled')} & "
        f"{_mean_std(row, 'temperature')}\\\\"
    )
