from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.db import connect_db
from imbalance_benchmark.analysis.query import load_runs_frame
from imbalance_benchmark.common import split_paths

__all__ = ["cost_comparison_rows", "write_cost_comparison_table"]

_IDENTITY_COLUMNS = {
    "patient_split",
    "assignment",
    "condition",
    "method",
    "seed_index",
}


def _paired_distribution(
    matched: pd.DataFrame, endpoint: str, n_replicates: int, seed: int
) -> np.ndarray:
    split_distributions = []
    for patient_split, group in matched.groupby("patient_split", sort=True):
        effects = (
            group[f"{endpoint}_method"] - group[f"{endpoint}_reference"]
        ).to_numpy()
        rng = np.random.default_rng(seed + int(str(patient_split)))
        indices = rng.integers(0, len(effects), size=(n_replicates, len(effects)))
        split_distributions.append(
            np.concatenate(([effects.mean()], effects[indices].mean(axis=1)))
        )
    if len(split_distributions) != 3:
        raise RuntimeError("Cost comparisons require all three patient splits")
    return np.mean(np.stack(split_distributions), axis=0)


def _matched_reference(
    frame: pd.DataFrame, assignment: str, condition: str, method: str
) -> pd.DataFrame:
    identity = ["patient_split", "seed_index"]
    method_rows = frame[
        (frame["assignment"] == assignment)
        & (frame["condition"] == condition)
        & (frame["method"] == method)
    ]
    reference = frame[
        (frame["assignment"] == assignment)
        & (frame["condition"] == condition)
        & (frame["method"] == "ce")
    ]
    endpoints = sorted(set(frame.columns) - _IDENTITY_COLUMNS)
    columns = [
        f"{name}_{role}" for name in endpoints for role in ("method", "reference")
    ]
    return cast(
        pd.DataFrame,
        method_rows.merge(
            reference,
            on=identity,
            suffixes=("_method", "_reference"),
            validate="one_to_one",
        )[[*identity, *columns]],
    )


def cost_comparison_rows(
    frame: pd.DataFrame, n_replicates: int, seed: int
) -> list[dict[str, object]]:
    """Compare every mitigation cost with matched imbalanced CE and retain a 95% CI."""
    endpoints = sorted(set(frame.columns) - _IDENTITY_COLUMNS)
    comparisons = frame.loc[
        frame["method"] != "ce", ["assignment", "condition", "method"]
    ]
    rows = []
    for assignment, condition, method in comparisons.drop_duplicates().itertuples(
        index=False, name=None
    ):
        matched = _matched_reference(frame, assignment, condition, method)
        if matched.empty:
            raise RuntimeError(
                f"Missing matched CE cost reference for {assignment}/{condition}/{method}"
            )
        rows.extend(
            _endpoint_rows(
                matched,
                endpoints,
                (assignment, condition, method),
                n_replicates,
                seed,
            )
        )
    return rows


def _endpoint_rows(
    matched: pd.DataFrame,
    endpoints: list[str],
    identity: tuple[str, str, str],
    n_replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    assignment, condition, method = identity
    rows = []
    for endpoint in endpoints:
        distribution = _paired_distribution(matched, endpoint, n_replicates, seed)
        low, high = np.percentile(distribution[1:], [2.5, 97.5])
        rows.append(
            {
                "assignment": assignment,
                "condition": condition,
                "method": method,
                "reference": f"{condition}/ce",
                "endpoint": endpoint,
                "effect": float(distribution[0]),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return rows


def _load_cost_frame(base_paths: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        conn = connect_db(paths["db"])
        try:
            runs = load_runs_frame(conn)
        finally:
            conn.close()
        costs = pd.json_normalize(
            [json.loads(raw) if raw else {} for raw in runs["cost_json"]]
        )
        identity = cast(
            pd.DataFrame,
            runs[["assignment", "condition", "method", "seed_index"]].reset_index(
                drop=True
            ),
        )
        frames.append(pd.concat([identity, costs], axis=1).assign(patient_split=index))
    return pd.concat(frames, ignore_index=True)


def write_cost_comparison_table(
    base_paths: dict[str, Path], n_replicates: int, seed: int
) -> None:
    """Write matched computational-cost effects and paired seed-bootstrap intervals."""
    table = pd.DataFrame(
        cost_comparison_rows(_load_cost_frame(base_paths), n_replicates, seed)
    )
    body = (
        "\\multicolumn{1}{c}{No matched cost comparisons available.}"
        if table.empty
        else table.to_latex(index=False, float_format="%.3f", escape=True)
    )
    text = (
        "\\begin{table}[ht]\n\\centering\n"
        f"{body}\n"
        "\\caption{Matched computational-cost effects with paired 95\\% confidence intervals.}\n"
        "\\label{tab:cost-comparisons}\n\\end{table}\n"
    )
    base_paths["tables"].mkdir(parents=True, exist_ok=True)
    (base_paths["tables"] / "cost_comparison_intervals.tex").write_text(
        text, encoding="utf-8"
    )
