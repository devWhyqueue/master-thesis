from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _tuning_table_column_spec() -> str:
    return (
        "@{}"
        ">{\\raggedright\\arraybackslash}p{0.08\\linewidth}"
        ">{\\raggedright\\arraybackslash}p{0.20\\linewidth}"
        ">{\\raggedright\\arraybackslash}X"
        ">{\\centering\\arraybackslash}p{0.09\\linewidth}"
        ">{\\centering\\arraybackslash}p{0.08\\linewidth}"
        ">{\\centering\\arraybackslash}p{0.09\\linewidth}"
        ">{\\centering\\arraybackslash}p{0.10\\linewidth}"
        "@{}"
    )


def write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    """Write the validation-tuning selection table for the report."""
    lines = [
        f"\\begin{{tabularx}}{{\\linewidth}}{{{_tuning_table_column_spec()}}}",
        "\\toprule",
        "Regime & Method & Selected params & Fixed F1 & Val F1 & Tuned F1 & Bal.\\ acc.\\\\",
        "\\midrule",
    ]
    lines.extend(_latex_rows(frame))
    lines.extend(["\\bottomrule", "\\end{tabularx}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_delta_figure(frame: pd.DataFrame, path: Path) -> None:
    """Plot tuned test macro-F1 deltas against the fixed CE baseline."""
    plot_frame = frame[frame["selected_params"] != "fixed baseline"]
    if plot_frame.empty:
        return
    rows = [dict(row) for _, row in plot_frame.iterrows()]
    rows.sort(
        key=lambda row: (str(row["benchmark"]), float(row["tuned_delta_macro_f1"]))
    )
    labels = [f"{row['regime']}: {row['method_label']}" for row in rows]
    values = [float(row["tuned_delta_macro_f1"]) for row in rows]
    colors = ["#4c9a6a" if value >= 0 else "#c44e52" for value in values]
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel(r"Tuned test $\Delta$ macro F1 vs. CE / MIL CE")
    ax.set_title("Validation-selected tuning robustness check")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_empty_outputs(paths: dict[str, Path]) -> None:
    """Write pending placeholders when no tuning outputs exist yet."""
    columns = [
        "benchmark",
        "regime",
        "method",
        "method_label",
        "selected_params",
        "tuned_val_macro_f1",
        "tuned_test_macro_f1",
        "tuned_test_balanced_accuracy",
    ]
    pd.DataFrame(columns=pd.Index(columns)).to_csv(
        paths["tables"] / "result_tuning_selection.csv", index=False
    )
    (paths["tables"] / "result_tuning_selection.tex").write_text(
        "\\emph{Validation-tuning results are pending.}\n", encoding="utf-8"
    )


def _latex_rows(frame: pd.DataFrame) -> list[str]:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"{row['regime']} & {row['method_label']} & "
            f"{_format_params(str(row['selected_params']))} & "
            f"{_format_optional(row['fixed_test_macro_f1'])} & "
            f"{_format_optional(row['tuned_val_macro_f1'])} & "
            f"{_format_optional(row['tuned_test_macro_f1'])} & "
            f"{_format_optional(row['tuned_test_balanced_accuracy'])}\\\\"
        )
    return rows


def _format_params(params_json: str) -> str:
    if params_json == "fixed baseline":
        return params_json
    params = json.loads(params_json)
    return ", ".join(
        f"{_latex_escape(key)}={float(value):g}" for key, value in params.items()
    )


def _format_optional(value: Any) -> str:
    if pd.isna(value):
        return "--"
    return f"\\num{{{float(value):.3f}}}"


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")
