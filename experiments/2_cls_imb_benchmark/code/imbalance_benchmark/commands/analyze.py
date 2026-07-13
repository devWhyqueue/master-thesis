from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.analysis.calibration import reliability_curve
from imbalance_benchmark.analysis.db import connect_db, init_schema
from imbalance_benchmark.analysis.inference.bootstrap import bootstrap_preflight
from imbalance_benchmark.analysis.inference.holm import apply_holm
from imbalance_benchmark.analysis.inference.recovery import gates_and_recovery
from imbalance_benchmark.analysis.pipeline import calibration_summary, ingest_all_runs
from imbalance_benchmark.analysis.query import load_classwise, load_seed_predictions
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
    manifest_path = paths["data"] / "manifest.csv"
    if manifest_path.exists():
        df_test = pd.read_csv(manifest_path)
        test_rows = cast(pd.DataFrame, df_test[df_test["split"] == "test"])
        write_json(
            paths["data"] / "bootstrap_preflight.json",
            bootstrap_preflight(test_rows, n_replicates, seed),
        )
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


def _aggregate_split_comparisons(base_paths: dict[str, Path]) -> None:
    """Equal-weight average split-specific estimands without duplicating assignments."""
    rows = []
    for index in range(3):
        path = split_paths(base_paths, index) / "data" / "gates_and_recovery.json"
        if not path.exists():
            continue
        for comparison in json.loads(path.read_text()).get("comparisons", []):
            rows.append({**comparison, "patient_split": index})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    keys = [key for key in ("assignment", "severity", "method", "gate") if key in frame]
    grouped = frame.groupby(keys, dropna=False)
    aggregate = []
    for key, group in grouped:
        entry = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        effects = group["effect"].dropna()
        entry.update(
            {
                "effect": float(effects.mean()) if not effects.empty else None,
                "n_splits": int(group["patient_split"].nunique()),
                "split_effects": {
                    str(row.patient_split): row.effect
                    for row in group[["patient_split", "effect"]].itertuples(index=False)
                },
            }
        )
        aggregate.append(entry)
    write_json(base_paths["data"] / "cross_split_gates_and_recovery.json", {"comparisons": aggregate})


def cmd_analyze(args: argparse.Namespace) -> None:
    """Rebuild the result database, calibration/gate/recovery diagnostics, tables, and figures."""
    if args.split_index is None:
        for index in range(3):
            cmd_analyze(argparse.Namespace(**vars(args), split_index=index))
        _aggregate_split_comparisons(ensure_dirs(load_config(args.config)))
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
