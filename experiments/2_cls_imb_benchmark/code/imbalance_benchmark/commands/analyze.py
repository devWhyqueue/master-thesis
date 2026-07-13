from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import pandas as pd
import numpy as np

from imbalance_benchmark.analysis.calibration import reliability_curve
from imbalance_benchmark.analysis.db import connect_db, init_schema
from imbalance_benchmark.analysis.inference.holm import apply_holm
from imbalance_benchmark.analysis.inference.gates import (
    calibration_gate,
    confidence_interval,
    discrimination_gate,
)
from imbalance_benchmark.analysis.inference.recovery import gates_and_recovery
from imbalance_benchmark.analysis.inference.crossed_permutation import (
    crossed_block_permutation_ba,
    crossed_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.metrics import assign_tiers
from imbalance_benchmark.analysis.pipeline import calibration_summary, ingest_all_runs
from imbalance_benchmark.analysis.query import (
    load_classwise,
    load_seed_predictions,
    load_test_identity,
)
from imbalance_benchmark.analysis.reporting.plots import (
    plot_reliability_diagram,
    plot_tail_vs_support,
)
from imbalance_benchmark.analysis.reporting.tables import (
    calibration_table,
    confirmatory_table,
    results_table,
)
from imbalance_benchmark.common import ensure_dirs, load_config, split_paths, write_json

__all__ = ["cmd_analyze"]


def _load_freeze(paths: dict[str, Path]) -> dict[str, Any]:
    """Load the frozen analysis manifest, if `freeze` has already produced one."""
    freeze_path = paths["data"] / "manifest_freeze.json"
    return json.loads(freeze_path.read_text()) if freeze_path.exists() else {}


def _write_diagnostics(
    paths: dict[str, Path],
    comparisons: list[dict[str, Any]],
    n_replicates: int,
    seed: int,
) -> None:
    """Write the bootstrap preflight, gate/recovery, and calibration diagnostic JSON files."""
    write_json(
        paths["data"] / "gates_and_recovery.json",
        {"comparisons": apply_holm(comparisons)},
    )
    write_json(paths["data"] / "calibration_summary.json", calibration_summary(paths))


def _write_tables(
    paths: dict[str, Path], conn: sqlite3.Connection, comparisons: list[dict[str, Any]]
) -> None:
    """Replace the placeholder LaTeX tables with real DB-driven results."""
    paths["tables"].mkdir(parents=True, exist_ok=True)
    (paths["tables"] / "results_table.tex").write_text(
        results_table(conn), encoding="utf-8"
    )
    (paths["tables"] / "calibration_table.tex").write_text(
        calibration_table(conn), encoding="utf-8"
    )
    (paths["tables"] / "confirmatory_table.tex").write_text(
        confirmatory_table(apply_holm(comparisons)), encoding="utf-8"
    )


def _write_figures(paths: dict[str, Path], conn: sqlite3.Connection) -> None:
    """Replace the placeholder figure with real tail-vs-support and reliability plots."""
    paths["figures"].mkdir(parents=True, exist_ok=True)
    classwise = load_classwise(conn)
    classwise_test = cast(pd.DataFrame, classwise[classwise["split"] == "test"])
    if not classwise_test.empty:
        plot_tail_vs_support(classwise_test, paths["figures"] / "tail_vs_support.png")
    balanced = load_seed_predictions(paths, "balanced", "ce")
    if balanced is not None:
        centers, mean_conf, accuracy = reliability_curve(
            balanced["probs"][0], balanced["labels"]
        )
        if len(centers):
            plot_reliability_diagram(
                centers,
                mean_conf,
                accuracy,
                paths["figures"] / "reliability_diagram.png",
            )


def _crossed_p_value(
    entry: dict[str, Any],
    base_paths: dict[str, Path],
    config: dict[str, Any],
    seed: int,
) -> float | None:
    """Calculate the gate statistic's one shared-block permutation p-value across splits."""
    if entry["method"] == "ce" or not entry.get("gate_passed"):
        return None
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    blocks = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        method = load_seed_predictions(
            paths, entry["severity"], entry["method"], entry["assignment"]
        )
        ce = load_seed_predictions(paths, entry["severity"], "ce", entry["assignment"])
        if method is None or ce is None:
            return None
        identity = load_test_identity(paths["data"] / "manifest.csv", is_mil)
        values = (
            method["preds"] if entry["gate"] == "discrimination" else method["probs"]
        )
        reference = ce["preds"] if entry["gate"] == "discrimination" else ce["probs"]
        blocks.append(
            (method["labels"], values, reference, identity["case_id"].to_numpy())
        )
    if entry["gate"] == "discrimination":
        return crossed_block_permutation_ba(
            blocks, len(method["class_names"]), seed=seed
        )
    freeze = _load_freeze(split_paths(base_paths, 0))
    allocated = freeze["assignment_conditions"][entry["assignment"]]["severe"][
        "allocated_counts"
    ]
    tail_classes = [
        index
        for index, name in enumerate(method["class_names"])
        if assign_tiers(method["class_names"], allocated).get(name) == "tail"
    ]
    return crossed_block_permutation_tail_nll(blocks, tail_classes, seed=seed)


def _aggregate_split_comparisons(
    base_paths: dict[str, Path], config: dict[str, Any] | None = None, seed: int = 0
) -> None:
    """Recompute crossed, equal-split effects within each shared bootstrap replicate."""
    rows = []
    for index in range(3):
        path = split_paths(base_paths, index)["data"] / "gates_and_recovery.json"
        if not path.exists():
            continue
        for comparison in json.loads(path.read_text()).get("comparisons", []):
            rows.append({**comparison, "patient_split": index})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    keys = [key for key in ("assignment", "severity", "method", "gate") if key in frame]
    grouped = frame.groupby(keys, dropna=False)
    aggregate: list[dict[str, Any]] = []
    for key, group in grouped:
        entry = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        effect_dist = np.mean(
            np.stack(group["bootstrap_effect"].map(np.asarray).tolist()), axis=0
        )
        entry.update(
            {
                "effect": float(np.nanmean(effect_dist)),
                "ci": confidence_interval(effect_dist),
                "n_splits": int(group["patient_split"].nunique()),
                "split_effects": {
                    str(row.patient_split): row.effect
                    for row in group[["patient_split", "effect"]].itertuples(
                        index=False
                    )
                },
            }
        )
        if (
            "bootstrap_numerator" in group
            and group["bootstrap_numerator"].notna().all()
        ):
            numerator = np.mean(
                np.stack(group["bootstrap_numerator"].map(np.asarray).tolist()), axis=0
            )
            denominator = np.mean(
                np.stack(group["bootstrap_denominator"].map(np.asarray).tolist()),
                axis=0,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                recovery = np.where(denominator != 0, numerator / denominator, np.nan)
            entry["recovery"] = float(np.nanmean(recovery))
            entry["recovery_ci"] = confidence_interval(recovery)
        aggregate.append(entry)
    gate_lookup = {
        (entry["assignment"], entry["severity"], entry["gate"]): entry
        for entry in aggregate
        if entry["method"] == "ce"
    }
    for entry in aggregate:
        gate = gate_lookup.get((entry["assignment"], entry["severity"], entry["gate"]))
        if gate is None:
            continue
        entry["gate_passed"] = (
            discrimination_gate(entry["effect"], entry["ci"])
            if entry["method"] == "ce" and entry["gate"] == "discrimination"
            else calibration_gate(entry["effect"], entry["ci"])
            if entry["method"] == "ce"
            else gate["gate_passed"]
        )
        entry["p_value"] = (
            _crossed_p_value(entry, base_paths, config, seed) if config else None
        )
    write_json(
        base_paths["data"] / "cross_split_gates_and_recovery.json",
        {"comparisons": apply_holm(aggregate)},
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Rebuild the result database, calibration/gate/recovery diagnostics, tables, and figures."""
    if args.split_index is None:
        for index in range(3):
            cmd_analyze(argparse.Namespace(**vars(args), split_index=index))
        _aggregate_split_comparisons(
            ensure_dirs(load_config(args.config)), load_config(args.config), args.seed
        )
        return
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    freeze = _load_freeze(paths)
    n_replicates = int(config.get("analysis", {}).get("bootstrap_replicates", 10_000))
    seed = int(getattr(args, "seed", 0) or 0)

    conn = connect_db(paths["db"])
    init_schema(conn)
    ingest_all_runs(conn, paths, freeze)

    comparisons = gates_and_recovery(paths, config, freeze, n_replicates, seed)
    _write_diagnostics(paths, comparisons, n_replicates, seed)
    _write_tables(paths, conn, comparisons)
    _write_figures(paths, conn)
    conn.close()
