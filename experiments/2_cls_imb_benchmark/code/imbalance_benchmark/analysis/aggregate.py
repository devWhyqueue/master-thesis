from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.db import connect_db
from imbalance_benchmark.analysis.inference.gates import (
    calibration_gate,
    confidence_interval,
    discrimination_gate,
)
from imbalance_benchmark.analysis.inference.holm import apply_holm
from imbalance_benchmark.analysis.query import load_classwise, load_eval_details
from imbalance_benchmark.common import split_paths, write_json


def aggregate_split_comparisons(
    base_paths: dict[str, Path],
    config: dict[str, Any] | None,
    seed: int,
    crossed_p_value: Callable[
        [dict[str, Any], dict[str, Path], dict[str, Any], int], float | None
    ],
) -> None:
    """Recompute crossed, equal-split effects within each shared bootstrap replicate."""
    rows = _comparison_rows(base_paths)
    frame = pd.DataFrame(rows)
    keys = [key for key in ("assignment", "severity", "method", "gate") if key in frame]
    aggregate = [
        _aggregate_group(keys, key, group)
        for key, group in frame.groupby(keys, dropna=False)
    ]
    _apply_gates(aggregate, base_paths, config, seed, crossed_p_value)
    write_json(
        base_paths["data"] / "cross_split_gates_and_recovery.json",
        {"comparisons": apply_holm(aggregate)},
    )


def _comparison_rows(base_paths: dict[str, Path]) -> list[dict[str, Any]]:
    """Load the three required split-level gate/recovery outputs."""
    rows = []
    for index in range(3):
        path = split_paths(base_paths, index)["data"] / "gates_and_recovery.json"
        if path.exists():
            rows.extend(
                {**comparison, "patient_split": index}
                for comparison in json.loads(path.read_text()).get("comparisons", [])
            )
    if {row["patient_split"] for row in rows} != {0, 1, 2}:
        raise RuntimeError(
            "Exactly three completed patient splits are required for confirmatory aggregation"
        )
    return rows


def _aggregate_group(
    keys: list[str], key: object, group: pd.DataFrame
) -> dict[str, Any]:
    """Average one comparison over fixed split repetitions and bootstrap draws."""
    entry = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
    effects = np.mean(
        np.stack(group["bootstrap_effect"].map(np.asarray).tolist()), axis=0
    )
    entry.update(
        effect=float(np.nanmean(effects)),
        ci=confidence_interval(effects),
        descriptive_only=bool(group["descriptive_only"].any())
        if "descriptive_only" in group
        else False,
        n_splits=int(group["patient_split"].nunique()),
        split_effects={
            str(split): effect
            for split, effect in group[["patient_split", "effect"]].itertuples(
                index=False, name=None
            )
        },
    )
    if "bootstrap_numerator" in group and bool(
        group["bootstrap_numerator"].notna().all()
    ):
        _add_recovery(entry, group)
    return entry


def _add_recovery(entry: dict[str, Any], group: pd.DataFrame) -> None:
    """Recompute the recovery ratio inside each equal-split bootstrap draw."""
    numerator = np.mean(
        np.stack(group["bootstrap_numerator"].map(np.asarray).tolist()), axis=0
    )
    denominator = np.mean(
        np.stack(group["bootstrap_denominator"].map(np.asarray).tolist()), axis=0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        recovery = np.where(denominator != 0, numerator / denominator, np.nan)
    entry.update(
        recovery=float(np.nanmean(recovery)), recovery_ci=confidence_interval(recovery)
    )


def _apply_gates(
    aggregate: list[dict[str, Any]],
    base_paths: dict[str, Path],
    config: dict[str, Any] | None,
    seed: int,
    crossed_p_value: Callable[
        [dict[str, Any], dict[str, Path], dict[str, Any], int], float | None
    ],
) -> None:
    """Propagate CE gates and add crossed patient-block permutation p-values."""
    lookup = {
        (entry["assignment"], entry["severity"], entry["gate"]): entry
        for entry in aggregate
        if entry["method"] == "ce"
    }
    for entry in lookup.values():
        entry["gate_passed"] = (
            False
            if entry.get("descriptive_only")
            else (
                discrimination_gate(entry["effect"], entry["ci"])
                if entry["gate"] == "discrimination"
                else calibration_gate(entry["effect"], entry["ci"])
            )
        )
    for entry in aggregate:
        gate = lookup.get((entry["assignment"], entry["severity"], entry["gate"]))
        if gate is not None:
            entry["gate_passed"] = entry.get("gate_passed", gate["gate_passed"])
            entry["p_value"] = (
                crossed_p_value(entry, base_paths, config, seed)
                if config and not entry.get("descriptive_only")
                else None
            )


def write_equal_split_endpoint_table(base_paths: dict[str, Path]) -> None:
    """Write canonical endpoint summaries with equal, rather than run-count, split weight."""
    equal = _equal_split_endpoints(base_paths)
    base_paths["tables"].mkdir(parents=True, exist_ok=True)
    (base_paths["tables"] / "equal_split_endpoints.tex").write_text(
        _latex_endpoint_table(equal), encoding="utf-8"
    )
    classwise = _equal_split_classwise(base_paths)
    (base_paths["tables"] / "equal_split_classwise_endpoints.tex").write_text(
        _latex_classwise_table(classwise), encoding="utf-8"
    )


def _equal_split_classwise(base_paths: dict[str, Path]) -> pd.DataFrame:
    """Per-class and tier endpoints averaged first within, then equally across, splits.

    Per-class recall/F1/NLL/Brier are stored per run but were absent from the
    canonical equal-split summary; this restores them alongside their locked
    head/body/tail tier so classwise damage and recovery are reportable.
    """
    frames = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        if not paths["db"].exists():
            raise RuntimeError(
                "Every patient split must be analysed before aggregation"
            )
        conn = connect_db(paths["db"])
        try:
            frame = load_classwise(conn)
        finally:
            conn.close()
        frame["patient_split"] = index
        frames.append(frame[frame["split"] == "test"])
    details = pd.concat(frames, ignore_index=True)
    metrics = ["recall", "f1", "nll", "brier"]
    keys = ["assignment", "condition", "method", "class_name", "tier"]
    per_split = details.groupby([*keys, "patient_split"], as_index=False)[
        metrics
    ].mean()
    return cast(
        pd.DataFrame,
        per_split.groupby(keys, as_index=False)[metrics].mean(),
    )


def _latex_classwise_table(classwise: pd.DataFrame) -> str:
    """Render the canonical equal-split per-class/tier endpoint table as LaTeX."""
    return (
        "\\begin{table}[ht]\n\\centering\n"
        + classwise.to_latex(index=False, float_format="%.3f")
        + "\\caption{Equal-weight three-split per-class and tier test endpoints.}\n"
        "\\label{tab:equal-split-classwise}\n\\end{table}\n"
    )


def _equal_split_endpoints(base_paths: dict[str, Path]) -> pd.DataFrame:
    """Load test results and average first within, then equally across, splits."""
    frames = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        if not paths["db"].exists():
            raise RuntimeError(
                "Every patient split must be analysed before aggregation"
            )
        conn = connect_db(paths["db"])
        try:
            frame = load_eval_details(conn)
        finally:
            conn.close()
        frame["patient_split"] = index
        frames.append(frame[frame["split"] == "test"])
    details = pd.concat(frames, ignore_index=True)
    metrics = [
        "balanced_accuracy",
        "macro_f1",
        "negative_log_likelihood",
        "macro_nll",
        "brier_score",
        "expected_calibration_error",
        "quadratic_weighted_kappa",
        "ordinal_mean_absolute_error",
        "slide_macro_balanced_accuracy",
        "patient_macro_balanced_accuracy",
        "slide_macro_f1",
        "patient_macro_f1",
        "slide_macro_nll",
        "patient_macro_nll",
        "slide_macro_brier",
        "patient_macro_brier",
    ]
    metrics = [metric for metric in metrics if metric in details.columns]
    per_split = details.groupby(
        ["patient_split", "assignment", "condition", "method"], as_index=False
    )[metrics].mean()
    return cast(
        pd.DataFrame,
        per_split.groupby(["assignment", "condition", "method"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns="patient_split"),
    )


def _latex_endpoint_table(equal: pd.DataFrame) -> str:
    """Render the canonical equal-split endpoint table as LaTeX."""
    return (
        "\\begin{table}[ht]\n\\centering\n"
        + equal.to_latex(index=False, float_format="%.3f")
        + "\\caption{Equal-weight three-split test endpoints.}\n\\label{tab:equal-split-results}\n\\end{table}\n"
    )
