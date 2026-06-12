"""Plotting helpers."""

from pathlib import Path

import pandas as pd


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
