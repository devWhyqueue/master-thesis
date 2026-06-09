from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common import EXPERIMENT_ROOT, ensure_dirs, load_config, write_json
from scripts.analysis.tuning.grid import SEEDS
from scripts.analysis.tuning.paths import tuning_result_dir

COPIED_FILES = (
    "val_results.json",
    "test_results.json",
    "config.json",
    "activation_diagnostics.json",
)
BASELINE_METHOD = {"patch_feature": "patch_feature_ce", "wsi_bag": "mil_ce"}


def report_result_root(paths: dict[str, Path], benchmark: str) -> Path:
    """Return the report-facing result root for one benchmark."""
    if benchmark == "patch_feature":
        return paths["root"] / "outputs" / "results" / "patch_feature"
    return paths["root"] / "outputs" / "results_wsi_bag"


def materialize_selected_results(
    paths: dict[str, Path], selected: pd.DataFrame, seeds: tuple[int, ...] = SEEDS
) -> list[dict[str, Any]]:
    """Copy validation-selected tuning runs into the report result tree."""
    missing: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        benchmark = str(row["benchmark"])
        method = str(row["method"])
        variant = str(row.get("variant", "") or "")
        dst_root = report_result_root(paths, benchmark) / method
        if method == BASELINE_METHOD[benchmark] or not variant:
            _ensure_baseline_present(dst_root, seeds, missing, benchmark, method)
            continue
        for seed in seeds:
            src = tuning_result_dir(paths, benchmark, method, variant, seed)
            dst = dst_root / f"seed={seed}"
            if not _copy_run(src, dst):
                missing.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "variant": variant,
                        "seed": seed,
                        "source": str(src),
                    }
                )
    return missing


def regenerate_selected_reports(config_path: str | Path | None = None) -> None:
    """Rebuild report tables and figures from the materialized selected results."""
    config = load_config(config_path)
    paths = ensure_dirs(config)
    python = sys.executable
    commands = _report_commands(python)
    for command in commands:
        subprocess.run(command, cwd=EXPERIMENT_ROOT, check=True)
    write_json(
        paths["tables"] / "selected_report_regeneration.json",
        {"status": "completed", "commands": len(commands)},
    )


def _report_commands(python: str) -> list[list[str]]:
    return [
        [python, "-m", "scripts.analysis.report.recompute_tier_metrics"],
        [python, "-m", "scripts.analysis.report.aggregate", "--benchmark", "patch"],
        [python, "-m", "scripts.analysis.report.aggregate", "--benchmark", "wsi_bag"],
        [python, "-m", "scripts.analysis.report.calibration.table"],
        [python, "-m", "scripts.analysis.report.calibration.posthoc_table"],
        [python, "-m", "scripts.analysis.report.paired_delta_table"],
        [python, "-m", "scripts.analysis.report.figures", "--benchmark", "patch"],
        [python, "-m", "scripts.analysis.report.figures", "--benchmark", "wsi_bag"],
        [python, "-m", "scripts.analysis.classwise_difficulty"],
        [python, "-m", "scripts.analysis.report.calibration.audit"],
    ]


def _ensure_baseline_present(
    dst_root: Path,
    seeds: tuple[int, ...],
    missing: list[dict[str, Any]],
    benchmark: str,
    method: str,
) -> None:
    for seed in seeds:
        result_dir = dst_root / f"seed={seed}"
        if (result_dir / "test_results.json").exists():
            continue
        missing.append(
            {
                "benchmark": benchmark,
                "method": method,
                "variant": "",
                "seed": seed,
                "source": str(result_dir),
            }
        )


def _copy_run(source: Path, destination: Path) -> bool:
    if not (source / "test_results.json").exists():
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for name in COPIED_FILES:
        src_file = source / name
        if src_file.exists():
            shutil.copy2(src_file, destination / name)
    return True
