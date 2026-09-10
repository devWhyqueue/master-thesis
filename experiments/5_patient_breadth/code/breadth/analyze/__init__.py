"""Analysis stage entry point for the patient-breadth experiment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from decodability import exp2_split_paths
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.common import output_root, verify_signed_file

from breadth.analyze.contrasts import (
    collect_cell_distributions,
    compute_contrasts_and_surface,
)
from breadth.analyze.report import write_report

__all__ = ["run_analyze"]

logger = logging.getLogger(__name__)


def run_analyze(config: dict[str, Any]) -> Path:
    """Run full contrast and support surface analysis across the 3x3 grid."""
    exp2_p = exp2_split_paths(config, 0)
    freeze = load_freeze_meta(exp2_p)
    n_classes = len(freeze["class_names"])

    preflight_p = output_root(config) / "data" / "preflight.json"
    verify_signed_file(preflight_p)
    preflight = json.loads(preflight_p.read_text(encoding="utf-8"))
    cohort_iccs = preflight.get("cohort_iccs", {})

    logger.info("Collecting bootstrap distributions across grid and draws...")
    pooled_dists, dispersions = collect_cell_distributions(config, n_classes)

    logger.info("Computing contrasts and fitting support surface models...")
    results = compute_contrasts_and_surface(config, pooled_dists, dispersions)

    report_p = write_report(config, results, cohort_iccs)
    logger.info("Patient-breadth analysis complete: %s", report_p)
    return report_p

