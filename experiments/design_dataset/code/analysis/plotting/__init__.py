"""Plotting helpers shared across all report modules."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

DISPLAY_NAMES: dict[str, str] = {
    "patch_feature_balanced_sampler_ce": "Balanced sampling",
    "patch_feature_ce_soft_f1_balanced": "CE + soft F1",
    "patch_feature_ce_soft_mcc_balanced": "CE + soft MCC",
    "patch_feature_cfal": "CFAL",
    "patch_feature_focal": "Focal loss",
    "patch_feature_oko": "OKO",
    "patch_feature_progan_aug": "ProGAN augmentation",
    "patch_feature_weighted_ce": "Weighted CE",
    "mde_mil": "MDE-MIL",
    "mil_balanced_sampler_ce": "Balanced sampling",
    "mil_focal": "Focal loss",
    "mil_weighted_ce": "Weighted CE",
    "rankmix_mil": "RankMix",
    "sc_mil": "SC-MIL",
}

PATCH_ORDER = [
    "patch_feature_weighted_ce",
    "patch_feature_focal",
    "patch_feature_cfal",
    "patch_feature_ce_soft_f1_balanced",
    "patch_feature_ce_soft_mcc_balanced",
    "patch_feature_balanced_sampler_ce",
    "patch_feature_oko",
    "patch_feature_progan_aug",
]

WSI_ORDER = [
    "mil_weighted_ce",
    "mil_focal",
    "mil_balanced_sampler_ce",
    "rankmix_mil",
    "sc_mil",
    "mde_mil",
]


def _benchmark(method: str) -> str:
    wsi_tokens = ("mil", "rankmix", "sc_mil", "mde")
    return "wsi_bag" if any(token in method for token in wsi_tokens) else "patch"


def _method_label(method: str) -> str:
    return DISPLAY_NAMES.get(method, method.replace("_", "\\_"))


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


def _write_wide_table(
    path: Path, colspec: str, header_lines: list[str], rows: list[str]
) -> None:
    body = ["\\begin{tabular}{" + colspec + "}", "\\toprule"]
    body.extend(f"{h}\\\\" for h in header_lines)
    body.extend(["\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(body) + "\n")


# --- Calibration statistics (used by calibration.py) ---


def negative_log_likelihood(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute mean negative log likelihood."""
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(labels)), labels])))


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute multiclass Brier score."""
    one_hot = np.eye(probabilities.shape[1])[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """Compute expected calibration error."""
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


def fit_temperature(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Find temperature minimising NLL on a grid."""
    log_t = np.linspace(np.log(0.05), np.log(10.0), 160)
    losses = [
        negative_log_likelihood(
            labels, temperature_scale(probabilities, float(np.exp(v)))
        )
        for v in log_t
    ]
    return float(np.exp(log_t[int(np.argmin(losses))]))


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to a probability matrix."""
    clipped = np.clip(probabilities, 1e-12, 1.0)
    logits = np.log(clipped) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# --- Result table writers ---


def write_result_tables(frame: pd.DataFrame, tables_dir: Path) -> None:
    """Write patch and WSI result summary tables."""
    for benchmark, filename in [
        ("patch", "result_summary_patch.tex"),
        ("wsi_bag", "result_summary_wsi_bag.tex"),
    ]:
        part = (
            cast(pd.DataFrame, frame[frame["benchmark"] == benchmark])
            if not frame.empty
            else frame
        )
        write_result_table(part, tables_dir / filename)


def write_result_table(frame: pd.DataFrame, path: Path) -> None:
    """Write one benchmark result table with lambda as columns, BAcc and F1 per cell."""
    params = [0.8, 1.1, 1.3]
    fallback_header = "Method & " + " & ".join(
        f"BAcc ($\\lambda={p:.1f}$) & F1 ($\\lambda={p:.1f}$)" for p in params
    )
    if frame.empty:
        _write_unavailable(path, fallback_header)
        return
    by_method = _by_method_map(frame)
    method_order = PATCH_ORDER if any("patch" in m for m in by_method) else WSI_ORDER
    ordered = [m for m in method_order if m in by_method] + [
        m for m in by_method if m not in method_order
    ]
    colspec, header_lines = _result_colspec_headers(params)
    rows = [_result_method_row(m, by_method[m], params) for m in ordered]
    _write_wide_table(path, colspec, header_lines, rows)


def _by_method_map(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = ("balanced_accuracy", "macro_f1")
    agg = (
        frame.groupby(["method", "parameter"])
        .agg(**{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")})
        .reset_index()
    )
    return {
        str(m): cast(pd.DataFrame, agg[agg["method"] == m])
        for m in agg["method"].unique()
    }


def _result_colspec_headers(params: list[float]) -> tuple[str, list[str]]:
    colspec = "l" + "r" * (len(params) * 2)
    header_top = "Method & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{$\\lambda={p:.1f}$}}" for p in params
    )
    cmidrule = " ".join(
        f"\\cmidrule(lr){{{2 + i * 2}--{3 + i * 2}}}" for i in range(len(params))
    )
    header_sub = " & ".join([""] + ["BAcc & F1"] * len(params))
    return colspec, [header_top + f" \\\\ {cmidrule}", header_sub]


def _result_method_row(method: str, part: pd.DataFrame, params: list[float]) -> str:
    cells = [_method_label(method)]
    for p in params:
        p_row = part[part["parameter"] == p]
        if p_row.empty:
            cells += ["--", "--"]
        else:
            r = p_row.iloc[0]
            cells += [_mean_std(r, "balanced_accuracy"), _mean_std(r, "macro_f1")]
    return " & ".join(cells) + "\\\\"


# --- Calibration table helpers (used by calibration.py) ---


def calibration_headline_colspec_headers(params: list[float]) -> tuple[str, list[str]]:
    """Return colspec and two-row header for a lambda-as-columns ECE calibration table."""
    colspec = "l" + "r" * (len(params) * 2 + 1)
    header_top = (
        "Method & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\lambda={p:.1f}$}}" for p in params)
        + " & $T$"
    )
    cmidrule = " ".join(
        f"\\cmidrule(lr){{{2 + i * 2}--{3 + i * 2}}}" for i in range(len(params))
    )
    header_sub = " & ".join([""] + ["ECE & ECE+TS"] * len(params) + [""])
    return colspec, [header_top + f" \\\\ {cmidrule}", header_sub]


def calibration_t_cell(t_vals: list) -> str:
    """Format fitted temperature column cell as mean ± std across lambda columns."""
    valid = [r for r in t_vals if r is not None]
    if not valid:
        return "--"
    t_mean = float(np.mean([float(r["temperature_mean"]) for r in valid]))
    t_std = float(np.std([float(r["temperature_mean"]) for r in valid]))
    return f"\\num{{{t_mean:.2f}}} $\\pm$ \\num{{{t_std:.2f}}}"
