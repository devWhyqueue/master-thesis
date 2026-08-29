from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.aggregation.consistent_severity import (
    require_consistent_achieved_severity,
)
from imbalance_benchmark.analysis.inference.gates import (
    calibration_gate,
    confidence_interval,
    discrimination_gate,
)
from imbalance_benchmark.analysis.inference.confirmatory.holm import apply_holm
from imbalance_benchmark.analysis.reporting.completeness import expected_comparison_keys
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
    require_complete_split_comparisons(
        rows, expected_comparison_keys(base_paths, config)
    )
    require_consistent_achieved_severity(base_paths)
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


def require_complete_split_comparisons(
    rows: list[dict[str, Any]], expected: set[tuple[str, str, str, str]] | None = None
) -> None:
    """Require every comparison, not merely some result, on all three locked splits."""
    keys = ("assignment", "severity", "method", "gate")
    frame = pd.DataFrame(rows)
    observed = set(frame[list(keys)].itertuples(index=False, name=None))
    missing = (expected or set()) - observed
    if missing:
        raise RuntimeError(
            f"Expected comparisons are missing across all splits: {sorted(missing)}"
        )
    for key, group in frame.groupby(list(keys), dropna=False):
        splits = set(group["patient_split"])
        if splits != {0, 1, 2} or len(group) != 3:
            raise RuntimeError(
                f"Comparison {key} is incomplete across patient splits: {sorted(splits)}"
            )


def _aggregate_group(
    keys: list[str], key: object, group: pd.DataFrame
) -> dict[str, Any]:
    """Average one comparison over fixed split repetitions and bootstrap draws."""
    entry = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
    effects = np.mean(
        np.stack(group["bootstrap_effect"].map(np.asarray).tolist()), axis=0
    )
    entry.update(
        # Replicate 0 carries the observed cross-split effect (equal split
        # weight); replicates 1.. supply the percentile interval.
        effect=float(effects[0]),
        ci=confidence_interval(effects),
        bootstrap_effect=effects.tolist(),
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
    # Constant per unit (frozen by matching_record.json), carried through so
    # the aggregated entry can be looked up as a matched-vs-unmatched contrast.
    for column in ("dominant", "matched_methods", "unmatched_methods"):
        if column in group:
            entry[column] = group[column].iloc[0]
    if "bootstrap_numerator" in group and bool(
        group["bootstrap_numerator"].notna().all()
    ):
        _add_recovery(entry, group)
    return entry


def _add_recovery(entry: dict[str, Any], group: pd.DataFrame) -> None:
    """Aggregate recovery across splits: observed point (index 0) plus a bootstrap CI.

    The point estimate is the ratio of the split-averaged observed numerator and
    denominator (equal split weight), not the mean of per-replicate ratios.
    """
    numerator = np.mean(
        np.stack(group["bootstrap_numerator"].map(np.asarray).tolist()), axis=0
    )
    denominator = np.mean(
        np.stack(group["bootstrap_denominator"].map(np.asarray).tolist()), axis=0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        recovery = np.where(denominator != 0, numerator / denominator, np.nan)
    recovery_point = numerator[0] / denominator[0] if denominator[0] != 0 else np.nan
    entry.update(
        recovery=float(recovery_point),
        recovery_ci=confidence_interval(recovery),
        bootstrap_numerator=numerator.tolist(),
        bootstrap_denominator=denominator.tolist(),
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
    dataset = (config or {}).get("dataset", {}).get("name")
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
                discrimination_gate(entry["effect"], entry["ci"], dataset)
                if entry["gate"] == "discrimination"
                else calibration_gate(entry["effect"], entry["ci"], dataset)
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
