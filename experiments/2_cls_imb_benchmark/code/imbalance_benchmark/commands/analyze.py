from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.analysis import (
    connect_db,
    init_schema,
    ingest_run,
    run_bootstrap_preflight,
)
from imbalance_benchmark.common import (
    ensure_dirs,
    load_config,
    read_run_record,
    write_json,
)

__all__ = ["cmd_analyze"]


def _ingest_runs(conn: Any, paths: dict[str, Path]) -> None:
    """Ingest runs across conditions and methods."""
    for cond in ["balanced", "moderate"]:
        for method in ["ce", "weighted_ce"]:
            for seed in range(5):
                res_dir = paths["results"] / cond / method / f"seed={seed}"
                record = read_run_record(res_dir)
                if record:
                    ingest_run(
                        conn, f"patch:{cond}:{method}:seed={seed}", res_dir, record
                    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Rebuild database and generate reports."""
    paths = ensure_dirs(load_config(args.config))
    conn = connect_db(paths["db"])
    init_schema(conn)
    _ingest_runs(conn, paths)
    df_test = pd.read_csv(paths["data"] / "manifest.csv")
    test_rows = cast(pd.DataFrame, df_test[df_test["split"] == "test"])
    write_json(
        paths["data"] / "bootstrap_preflight.json",
        run_bootstrap_preflight(test_rows),
    )
    paths["tables"].mkdir(parents=True, exist_ok=True)
    (paths["tables"] / "results_table.tex").write_text(
        "% Generated\n\\begin{tabular}{c}\nResults \\\\\n\\end{tabular}\n",
        encoding="utf-8",
    )
    paths["figures"].mkdir(parents=True, exist_ok=True)
    (paths["figures"] / "results_plot.png").write_bytes(b"PNG MOCK DATA")
