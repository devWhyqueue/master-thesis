from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.db import connect_db
from imbalance_benchmark.analysis.query import load_eval_details
from imbalance_benchmark.common import split_paths


def _calibration_row(
    key: tuple[str, str, str], values: dict[str, np.ndarray]
) -> dict[str, object]:
    assignment, condition, method = key
    return {
        "assignment": assignment,
        "condition": condition,
        "method": method,
        **_distribution_summary(values["expected_calibration_error"], "ECE"),
        **_distribution_summary(
            values["temperature_scaled_expected_calibration_error"],
            "Temperature ECE",
        ),
    }


def write_crossed_calibration_table(
    base_paths: dict[str, Path],
    distributions: dict[tuple[str, str, str], dict[str, np.ndarray]],
) -> None:
    """Write the final ECE table from already-computed per-key endpoint distributions."""
    rows = [
        _calibration_row(key, values) for key, values in sorted(distributions.items())
    ]
    table = pd.DataFrame(rows)
    body = (
        "\\multicolumn{1}{c}{No confirmed runs ingested yet.}"
        if table.empty
        else table.to_latex(index=False, float_format="%.3f", escape=True)
    )
    text = (
        "% Raw and temperature-scaled calibration summary with crossed ECE intervals\n"
        "\\begin{table}[ht]\n\\centering\n"
        f"{body}\n"
        "\\caption{Calibration summary with crossed patient-bootstrap ECE intervals}\n"
        "\\label{tab:calibration}\n\\end{table}\n"
    )
    base_paths["tables"].mkdir(parents=True, exist_ok=True)
    (base_paths["tables"] / "calibration_table.tex").write_text(text, encoding="utf-8")


def _complete_result_keys(base_paths: dict[str, Path]) -> set[tuple[str, str, str]]:
    keys_by_split = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        conn = connect_db(paths["db"])
        try:
            details = load_eval_details(conn)
        finally:
            conn.close()
        test = cast(pd.DataFrame, details[details["split"] == "test"])
        key_frame = cast(pd.DataFrame, test[["assignment", "condition", "method"]])
        keys_by_split.append(set(key_frame.itertuples(index=False, name=None)))
    return set.intersection(*keys_by_split) if keys_by_split else set()


def _distribution_summary(values: np.ndarray, name: str) -> dict[str, object]:
    array = np.asarray(values)
    replicates = array[1:] if len(array) > 1 else array
    low, high = np.percentile(replicates, [2.5, 97.5])
    return {name: float(array[0]), f"{name} 95% CI": f"[{low:.3f}, {high:.3f}]"}
