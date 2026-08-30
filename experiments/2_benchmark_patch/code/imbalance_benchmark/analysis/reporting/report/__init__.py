"""Report-shaped LaTeX fragments for the patch-benchmark results report.

Reads frozen analysis artifacts only -- no bootstrap, no permutation, no model
fit -- and emits one fragment per float in ``3_benchmark_patch_results.tex``.
Each fragment is a bare tabular or longtable body: a single float in the report
spans all four datasets, so caption and label stay with the float and are not
duplicated here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from imbalance_benchmark.analysis.reporting.report import appendix, endpoints, rq1, rq2
from imbalance_benchmark.analysis.reporting.report.construction import realized_support
from imbalance_benchmark.analysis.reporting.report.sources import (
    Dataset,
    dataset_roots,
    load_dataset,
)

__all__ = ["write_report_tables"]

logger = logging.getLogger(__name__)

_DATASET_TABLES: dict[str, Callable[[list[Dataset]], str]] = {
    "signal_profiles": rq1.signal_profiles,
    "discrimination_deficit": rq1.discrimination_deficit,
    "calibration_deficit": rq1.calibration_deficit,
    "gate_routing": rq1.gate_routing,
    "tier_endpoints": rq1.tier_endpoints,
    "natural_anchor": rq1.natural_anchor,
    "confirmatory": rq2.confirmatory,
    "matched_contrast": rq2.matched_contrast,
    "matched_beta": rq2.matched_beta,
    "calibration_recovery": rq2.calibration_recovery,
    "roster_recovery": rq2.roster_recovery,
    "classwise_endpoints": endpoints.classwise_endpoints,
    "calibration_detail": endpoints.calibration_detail,
    "cost": endpoints.cost,
    "preflight_outcome": appendix.preflight_outcome,
    "completeness": appendix.completeness,
    "tuning_selections": appendix.tuning_selections,
    "per_split": appendix.per_split,
    "method_diagnostics": appendix.method_diagnostics,
}

_RQ3_TABLES: dict[str, Callable[[dict[str, Any]], str]] = {
    "rq3_models": appendix.rq3_models,
    "rq3_logo": appendix.rq3_logo,
}


def _write(destination: Path, name: str, fragment: str) -> None:
    (destination / f"{name}.tex").write_text(fragment, encoding="utf-8")
    logger.info("report-tables: wrote %s", destination / f"{name}.tex")


def write_report_tables(config: dict[str, Any], base_paths: dict[str, Path]) -> None:
    """Emit every fragment the results report inputs, into ``tables/report/``."""
    destination = base_paths["tables"] / "report"
    destination.mkdir(parents=True, exist_ok=True)
    datasets = [load_dataset(root) for root in dataset_roots(config)]
    for name, build in _DATASET_TABLES.items():
        _write(destination, name, build(datasets))
    _write(
        destination,
        "realized_support",
        realized_support(datasets, destination / "realized_condition_support.csv"),
    )
    rq3 = json.loads((base_paths["data"] / "cross_dataset_rq3.json").read_text())
    for name, build_rq3 in _RQ3_TABLES.items():
        _write(destination, name, build_rq3(rq3))
