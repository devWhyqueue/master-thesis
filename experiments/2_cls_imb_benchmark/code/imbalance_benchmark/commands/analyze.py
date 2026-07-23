from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.analysis.calibration import seed_averaged_reliability_curve
from imbalance_benchmark.analysis.aggregate import (
    aggregate_split_comparisons,
    write_equal_split_endpoint_table,
)
from imbalance_benchmark.analysis.db import connect_db, init_schema
from imbalance_benchmark.analysis.inference.recovery import gates_and_recovery
from imbalance_benchmark.analysis.inference.holm import apply_holm
from imbalance_benchmark.analysis.inference.crossed_permutation import (
    crossed_p_value,
    load_freeze,
)
from imbalance_benchmark.analysis.reporting.ingestion import (
    ingest_all_runs,
    write_diagnostics,
)
from imbalance_benchmark.analysis.predictors.rq3_analysis import (
    cross_dataset_rq3,
    load_rq3_cells,
    run_rq3,
)
from imbalance_benchmark.analysis.query import load_classwise, load_seed_predictions
from imbalance_benchmark.analysis.reporting.plots import (
    plot_reliability_diagram,
    plot_tail_vs_support,
    write_tail_reliability,
)
from imbalance_benchmark.analysis.reporting.tables import (
    calibration_table,
    confirmatory_table,
    rq3_table,
    results_table,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.report import (
    write_interval_tables,
)
from imbalance_benchmark.common import (
    ensure_dirs,
    load_config,
    output_root,
    split_paths,
    write_json,
)
from imbalance_benchmark.manifest.seeds import derive_seed

__all__ = ["cmd_analyze", "cmd_analyze_combine", "cmd_combine_rq3"]

logger = logging.getLogger(__name__)


def cmd_combine_rq3(args: argparse.Namespace) -> None:
    """Fit the combined cross-dataset RQ3 analysis over every listed dataset-regime."""
    config = load_config(args.config)
    roots = [Path(r) for r in config.get("rq3", {}).get("dataset_roots", [])] or [
        output_root(config)
    ]
    base_paths = ensure_dirs(config)
    write_json(
        base_paths["data"] / "cross_dataset_rq3.json",
        cross_dataset_rq3(load_rq3_cells(roots)),
    )


def _write_tables(
    paths: dict[str, Path],
    conn: sqlite3.Connection,
    comparisons: list[dict[str, Any]],
    rq3: dict[str, Any],
) -> None:
    """Replace the placeholder LaTeX tables with real DB-driven results."""
    paths["tables"].mkdir(parents=True, exist_ok=True)
    for name, text in [
        ("results_table.tex", results_table(conn)),
        ("calibration_table.tex", calibration_table(conn)),
        ("confirmatory_table.tex", confirmatory_table(apply_holm(comparisons))),
        ("rq3_table.tex", rq3_table(rq3["models"])),
    ]:
        (paths["tables"] / name).write_text(text, encoding="utf-8")


def _write_figures(
    paths: dict[str, Path], conn: sqlite3.Connection, freeze: dict[str, Any]
) -> None:
    """Replace the placeholder figure with real tail-vs-support and reliability plots."""
    paths["figures"].mkdir(parents=True, exist_ok=True)
    classwise = load_classwise(conn)
    classwise_test = cast(pd.DataFrame, classwise[classwise["split"] == "test"])
    if not classwise_test.empty:
        plot_tail_vs_support(
            classwise_test, freeze, paths["figures"] / "tail_vs_support.png"
        )
    balanced = load_seed_predictions(paths, "balanced", "ce")
    if balanced is not None:
        centers, mean_conf, accuracy = seed_averaged_reliability_curve(
            balanced["probs"], balanced["labels"]
        )
        if len(centers):
            plot_reliability_diagram(
                centers,
                mean_conf,
                accuracy,
                paths["figures"] / "reliability_diagram.png",
            )
        write_tail_reliability(paths, freeze, balanced)


def _aggregate_split_comparisons(
    base_paths: dict[str, Path], config: dict[str, Any] | None = None, seed: int = 0
) -> None:
    """Recompute crossed, equal-split effects within each shared bootstrap replicate."""
    aggregate_split_comparisons(base_paths, config, seed, crossed_p_value)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Rebuild the result database, calibration/gate/recovery diagnostics, tables, and figures."""
    if args.split_index is None:
        _analyze_all_splits(args)
        return
    _analyze_one_split(args)


def _analyze_all_splits(args: argparse.Namespace) -> None:
    """Analyze all valid split repetitions, then produce equal-split summaries."""
    config = load_config(args.config)
    base_paths = ensure_dirs(config)
    excluded = [
        i
        for i in range(3)
        if (split_paths(base_paths, i)["data"] / "confirmatory_exclusion.json").exists()
    ]
    if excluded:
        write_json(
            base_paths["data"] / "confirmatory_exclusion.json",
            {"excluded": True, "failed_splits": excluded},
        )
        return
    for index in range(3):
        start = time.monotonic()
        logger.info("analyze: split %d/3 start", index + 1)
        _analyze_one_split(argparse.Namespace(**{**vars(args), "split_index": index}))
        logger.info(
            "analyze: split %d/3 done in %.1fs", index + 1, time.monotonic() - start
        )
    _aggregate_all_splits(args)


def _aggregate_all_splits(args: argparse.Namespace) -> None:
    """Cross-split exclusion guard, then equal-split aggregation, tables, and endpoints.

    Self-contained (re-scans exclusion) so it also works as the standalone
    Hydra `analyze-combine` job, which has no in-process knowledge of whether
    `_analyze_one_split` already ran for each split.
    """
    config = load_config(args.config)
    base_paths = ensure_dirs(config)
    excluded = [
        i
        for i in range(3)
        if (split_paths(base_paths, i)["data"] / "confirmatory_exclusion.json").exists()
    ]
    if excluded:
        write_json(
            base_paths["data"] / "confirmatory_exclusion.json",
            {"excluded": True, "failed_splits": excluded},
        )
        return
    logger.info("analyze: aggregating splits")
    _aggregate_split_comparisons(
        base_paths, config, derive_seed(args.seed, "resampling")
    )
    logger.info("analyze: interval tables")
    write_interval_tables(
        base_paths,
        config,
        int(config.get("analysis", {}).get("bootstrap_replicates", 10_000)),
        derive_seed(args.seed, "resampling"),
    )
    logger.info("analyze: equal-split endpoint table")
    write_equal_split_endpoint_table(base_paths)
    logger.info("analyze: aggregation done")


def cmd_analyze_combine(args: argparse.Namespace) -> None:
    """Aggregate the three split analyses into equal-split summaries (Hydra fan-in job)."""
    _aggregate_all_splits(args)


def _analyze_one_split(args: argparse.Namespace) -> None:
    """Ingest one split's runs and write its diagnostics, tables, and figures."""
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    if (paths["data"] / "confirmatory_exclusion.json").exists():
        return
    freeze = load_freeze(paths)
    n_replicates = int(config.get("analysis", {}).get("bootstrap_replicates", 10_000))
    seed = derive_seed(int(getattr(args, "seed", 0) or 0), "resampling")
    conn = connect_db(paths["db"])
    init_schema(conn)
    ingest_all_runs(conn, paths, freeze)
    comparisons = gates_and_recovery(paths, config, freeze, n_replicates, seed)
    rq3 = run_rq3(paths, config, freeze, comparisons)
    write_diagnostics(paths, comparisons)
    write_json(paths["data"] / "rq3.json", rq3)
    _write_tables(paths, conn, comparisons, rq3)
    _write_figures(paths, conn, freeze)
    conn.close()
