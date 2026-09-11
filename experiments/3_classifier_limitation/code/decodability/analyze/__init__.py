"""Analysis stage entry point for decodability experiment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.query import load_test_identity
from imbalance_benchmark.common import (
    ensure_dirs,
    output_root,
    split_paths,
    verify_signed_file,
)

from decodability import SUPPORTS, exp2_split_paths
from decodability.analyze.contrasts import compute_contrasts, load_readout_predictions
from decodability.analyze.endpoints import compute_cell_endpoints
from decodability.analyze.report import write_report
from decodability.evidence import load_freeze_meta

__all__ = ["run_analyze"]

logger = logging.getLogger(__name__)


def _gather_split_endpoints(
    config: dict[str, Any], selection: dict[str, Any], n_classes: int
) -> dict[str, Any]:
    """Gather primary and secondary endpoints for each split x support x readout."""
    res: dict[str, Any] = {}
    for split_index in range(3):
        paths = split_paths(ensure_dirs(config), split_index)
        exp2_paths = exp2_split_paths(config, split_index)
        identity = load_test_identity(
            exp2_paths["data"] / "manifest.csv", is_mil=False, split_name="test"
        )
        split_res: dict[str, Any] = {}
        for support in SUPPORTS:
            supp_res: dict[str, Any] = {}
            for readout in ("mlp", "logreg", "knn"):
                labels, preds_stack = load_readout_predictions(
                    config, split_index, support, readout, selection
                )
                preds = preds_stack[0]
                n_samples = len(labels)
                probs = np.zeros((n_samples, n_classes), dtype=np.float64)
                probs[np.arange(n_samples), preds] = 1.0
                ep = compute_cell_endpoints(labels, preds, probs, identity, n_classes)
                supp_res[readout] = ep
            split_res[support] = supp_res
        res[str(split_index)] = split_res
    return res


def run_analyze(config: dict[str, Any]) -> Path:
    """Run full contrast and endpoint analysis, producing json, tables, and figures."""
    root_p = output_root(config)
    sel_p = root_p / "data" / "probe_selection.json"
    verify_signed_file(sel_p)
    selection = json.loads(sel_p.read_text(encoding="utf-8"))

    exp2_p = exp2_split_paths(config, 0)
    freeze = load_freeze_meta(exp2_p)
    n_classes = len(freeze["class_names"])

    split_endpoints = _gather_split_endpoints(config, selection, n_classes)
    contrasts = compute_contrasts(config, selection, n_classes)

    report_p = write_report(config, contrasts, selection, split_endpoints)
    logger.info("Analysis report written to %s", report_p)
    return report_p
