import json
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

RESULT_PATTERN = re.compile(
    r"results_(?P<method>.+)/order=(?P<order>.+)/param=(?P<parameter>[\d.]+)/seed=(?P<seed>\d+)"
)
CALIBRATION_METRICS = (
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
)


def calibration_summary(root: Path) -> pd.DataFrame:
    """Return raw and temperature-scaled calibration metrics per run."""
    rows = []
    for test_path in sorted(
        root.glob("results_*/order=*/param=*/seed=*/test_results.json")
    ):
        match = RESULT_PATTERN.search(str(test_path.parent))
        validation_path = test_path.parent / "validation_results.json"
        if match is None or not validation_path.exists():
            continue
        row = _calibration_row(match.groupdict(), validation_path, test_path)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


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
    grouped = frame.groupby(["method", "order", "parameter"]).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in _raw_metrics()},
        **{f"{metric}_std": (metric, "std") for metric in _raw_metrics()},
        **{
            f"{metric}_scaled_mean": (f"{metric}_scaled", "mean")
            for metric in CALIBRATION_METRICS
        },
        **{
            f"{metric}_scaled_std": (f"{metric}_scaled", "std")
            for metric in CALIBRATION_METRICS
        },
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
    log_temperatures = np.linspace(np.log(0.05), np.log(10.0), 160)
    losses = [
        _negative_log_likelihood(
            labels, _temperature_scale(probabilities, float(np.exp(value)))
        )
        for value in log_temperatures
    ]
    return float(np.exp(log_temperatures[int(np.argmin(losses))]))


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


def _benchmark(method: str) -> str:
    wsi_tokens = ("mil", "rankmix", "sc_mil", "mde")
    return "wsi_bag" if any(token in method for token in wsi_tokens) else "patch"


def _mean_std(row: pd.Series, metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std = row[f"{metric}_std"]
    std_value = 0.0 if bool(pd.isna(std)) else float(std)
    return f"\\num{{{mean:.3f}}} $\\pm$ \\num{{{std_value:.3f}}}"


def _tex(value: object) -> str:
    return str(value).replace("_", "\\_")


def _write_unavailable(path: Path, header: str) -> None:
    columns = header.count("&") + 1
    row = f"\\multicolumn{{{columns}}}{{c}}{{Generated results unavailable.}}\\\\"
    _write_table(path, header, [row])


def _write_table(path: Path, header: str, rows: list[str]) -> None:
    spec = "l" * (header.count("&") + 1)
    body = ["\\begin{tabular}{" + spec + "}", "\\toprule", f"{header}\\\\"]
    body.extend(["\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(body) + "\n")
