from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _provenance_table_column_spec() -> str:
    return (
        "@{}"
        ">{\\raggedright\\arraybackslash}p{0.10\\linewidth}"
        ">{\\raggedright\\arraybackslash}p{0.22\\linewidth}"
        ">{\\raggedright\\arraybackslash}X"
        ">{\\centering\\arraybackslash}p{0.10\\linewidth}"
        ">{\\centering\\arraybackslash}p{0.10\\linewidth}"
        ">{\\centering\\arraybackslash}p{0.12\\linewidth}"
        "@{}"
    )


def write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    """Write the validation-selected configuration provenance table."""
    lines = [
        f"\\begin{{tabularx}}{{\\linewidth}}{{{_provenance_table_column_spec()}}}",
        "\\toprule",
        "Regime & Method & Selected control & Val macro F1 & Test macro F1 & Test bal.\\ acc.\\\\",
        "\\midrule",
    ]
    lines.extend(_latex_rows(frame))
    lines.extend(["\\bottomrule", "\\end{tabularx}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_empty_outputs(paths: dict[str, Path]) -> None:
    """Write pending placeholders when no tuning outputs exist yet."""
    columns = [
        "benchmark",
        "regime",
        "method",
        "method_label",
        "variant",
        "selected_params",
        "val_macro_f1",
        "test_macro_f1",
        "test_balanced_accuracy",
    ]
    pd.DataFrame(columns=pd.Index(columns)).to_csv(
        paths["tables"] / "result_tuning_selection.csv", index=False
    )
    (paths["tables"] / "result_tuning_selection.tex").write_text(
        "\\emph{Validation-selected configurations are pending.}\n", encoding="utf-8"
    )


def _latex_rows(frame: pd.DataFrame) -> list[str]:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"{row['regime']} & {row['method_label']} & "
            f"{_format_params(str(row['selected_params']))} & "
            f"{_format_optional(row['val_macro_f1'])} & "
            f"{_format_optional(row['test_macro_f1'])} & "
            f"{_format_optional(row['test_balanced_accuracy'])}\\\\"
        )
    return rows


def _format_params(params_json: str) -> str:
    if params_json in {"{}", "fixed baseline"}:
        return "baseline (no sweep)"
    params = json.loads(params_json)
    if not params:
        return "baseline (no sweep)"
    return ", ".join(
        f"{_latex_escape(key)}={float(value):g}" for key, value in params.items()
    )


def _format_optional(value: Any) -> str:
    if pd.isna(value):
        return "--"
    return f"\\num{{{float(value):.3f}}}"


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")
