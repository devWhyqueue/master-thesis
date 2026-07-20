from __future__ import annotations

import argparse
import json
import sqlite3
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
    crossed_block_permutation_ba,
    crossed_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.metrics import assign_tiers
from imbalance_benchmark.analysis.reporting.ingestion import (
    ingest_all_runs,
    write_diagnostics,
)
from imbalance_benchmark.analysis.predictors.rq3_analysis import (
    cross_dataset_rq3,
    load_rq3_cells,
    run_rq3,
)
from imbalance_benchmark.analysis.query import (
    load_classwise,
    load_seed_predictions,
    load_test_identity,
)
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
from imbalance_benchmark.analysis.reporting.calibration_intervals import (
    write_crossed_calibration_table,
)
from imbalance_benchmark.common import (
    ensure_dirs,
    load_config,
    output_root,
    split_paths,
    write_json,
)
from imbalance_benchmark.manifest.seeds import derive_seed

__all__ = ["cmd_analyze", "cmd_combine_rq3"]


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


def _load_freeze(paths: dict[str, Path]) -> dict[str, Any]:
    """Load the frozen analysis manifest, if `freeze` has already produced one."""
    freeze_path = paths["data"] / "manifest_freeze.json"
    return json.loads(freeze_path.read_text()) if freeze_path.exists() else {}


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
    method_data: dict[str, Any] | None = None
    for index in range(3):
        paths = split_paths(base_paths, index)
        method = load_seed_predictions(
            paths, entry["severity"], entry["method"], entry["assignment"]
        )
        ce = load_seed_predictions(paths, entry["severity"], "ce", entry["assignment"])
        if method is None or ce is None:
            return None
        method_data = method
        id_df = load_test_identity(paths["data"] / "manifest.csv", is_mil)
        is_disc = entry["gate"] == "discrimination"
        blocks.append(
            (
                method["labels"],
                method["preds"] if is_disc else method["probs"],
                ce["preds"] if is_disc else ce["probs"],
                id_df["case_id"].to_numpy(),
            )
        )
    if method_data is None:
        return None
    if entry["gate"] == "discrimination":
        return crossed_block_permutation_ba(
            blocks, len(method_data["class_names"]), seed=seed
        )
    tail_classes = []
    for i in range(3):
        fz = _load_freeze(split_paths(base_paths, i))
        alloc = fz["assignment_conditions"][entry["assignment"]][entry["severity"]][
            "allocated_counts"
        ]
        c_names = method_data["class_names"]
        tiers = assign_tiers(
            c_names,
            alloc,
            fz.get("tail_assignments", {}).get(entry["assignment"], c_names),
        )
        tail_classes.append(
            [idx for idx, name in enumerate(c_names) if tiers.get(name) == "tail"]
        )
    return crossed_block_permutation_tail_nll(blocks, tail_classes, seed=seed)


def _aggregate_split_comparisons(
    base_paths: dict[str, Path], config: dict[str, Any] | None = None, seed: int = 0
) -> None:
    """Recompute crossed, equal-split effects within each shared bootstrap replicate."""
    aggregate_split_comparisons(base_paths, config, seed, _crossed_p_value)


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
        _analyze_one_split(argparse.Namespace(**{**vars(args), "split_index": index}))
    _aggregate_split_comparisons(
        base_paths, config, derive_seed(args.seed, "resampling")
    )
    write_crossed_calibration_table(
        base_paths, config, derive_seed(args.seed, "resampling")
    )
    write_equal_split_endpoint_table(base_paths)


def _analyze_one_split(args: argparse.Namespace) -> None:
    """Ingest one split's runs and write its diagnostics, tables, and figures."""
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    if (paths["data"] / "confirmatory_exclusion.json").exists():
        return
    freeze = _load_freeze(paths)
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
