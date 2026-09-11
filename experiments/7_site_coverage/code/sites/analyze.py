"""Analyze stage: site gain, within-site residual breadth, and their interpretation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import output_root, write_json

from sites.classify import classify
from sites.fitting import fit, prepare
from sites.results import (
    allocation_payload,
    build_results,
    surface_payload,
    write_distributions,
)
from sites.strata import strata_analysis

__all__ = ["classify", "run_analyze"]

logger = logging.getLogger(__name__)


def run_analyze(config: dict[str, Any]) -> Path:
    """Compute the site gain, within-site residual, and their interpretation."""
    ctx = prepare(config)
    result = fit(config, ctx)
    beta_dist, gamma_dist, neff_deep, neff_broad, b_w, b_ref, delta_s = result.contrasts
    strata = strata_analysis(config, ctx.paths7, ctx.site_classes, ctx.class_names)
    results = build_results(
        ctx.site_classes,
        allocation_payload(result.dists, result.all_dists, result.dispersion),
        delta_s,
        b_w,
        b_ref,
        surface_payload(beta_dist, gamma_dist, neff_deep, neff_broad),
        strata,
    )
    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, results)
    write_distributions(
        config, result.dists, delta_s, b_w, b_ref, beta_dist, gamma_dist
    )
    logger.info("Site-coverage analysis complete: %s", out_p)
    return out_p
