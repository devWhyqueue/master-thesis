"""Analysis stage entry point for the patient-influence experiment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from decodability import exp2_split_paths
from decodability.analyze.endpoints import compute_cell_endpoints
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.analysis.query import load_test_identity
from imbalance_benchmark.common import output_root, verify_signed_file

from influence import OBJECTIVES, SUPPORTS, inherited_lambda
from influence.analyze.contrasts import compute_contrasts, load_objective_predictions
from influence.analyze.report import write_report

__all__ = ["run_analyze"]

logger = logging.getLogger(__name__)


def _gather_split_endpoints(config: dict[str, Any], n_classes: int) -> dict[str, Any]:
    """Gather primary and secondary endpoints for each split x support x objective."""
    res: dict[str, Any] = {}
    for split_index in range(3):
        exp2_paths = exp2_split_paths(config, split_index)
        identity = load_test_identity(
            exp2_paths["data"] / "manifest.csv", is_mil=False, split_name="test"
        )
        split_res: dict[str, Any] = {}
        for support in SUPPORTS:
            supp_res: dict[str, Any] = {}
            for objective in OBJECTIVES:
                labels, preds_stack = load_objective_predictions(
                    config, split_index, support, objective
                )
                preds = preds_stack[0]
                n_samples = len(labels)
                probs = np.zeros((n_samples, n_classes), dtype=np.float64)
                probs[np.arange(n_samples), preds] = 1.0
                supp_res[objective] = compute_cell_endpoints(
                    labels, preds, probs, identity, n_classes
                )
            split_res[support] = supp_res
        res[str(split_index)] = split_res
    return res


def _load_regularization(config: dict[str, Any]) -> dict[str, Any]:
    """Collect exp-3's inherited lambda for each support condition."""
    reg: dict[str, Any] = {}
    for support in SUPPORTS:
        lam, param_str = inherited_lambda(config, support)
        reg[support] = {"lambda": lam, "param": param_str}
    return reg


def _load_contribution_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Load the per-split, per-support contribution audit from the signed preflight."""
    preflight_path = output_root(config) / "data" / "preflight.json"
    verify_signed_file(preflight_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    return {
        split: {
            support: cell["contribution_audit"] for support, cell in supports.items()
        }
        for split, supports in preflight.get("splits", {}).items()
    }


def run_analyze(config: dict[str, Any]) -> Path:
    """Run full contrast and endpoint analysis, producing json, tables, and figures."""
    exp2_p = exp2_split_paths(config, 0)
    freeze = load_freeze_meta(exp2_p)
    n_classes = len(freeze["class_names"])

    split_endpoints = _gather_split_endpoints(config, n_classes)
    contrasts = compute_contrasts(config, n_classes)
    regularization = _load_regularization(config)
    contribution_audit = _load_contribution_audit(config)

    report_p = write_report(
        config, contrasts, regularization, contribution_audit, split_endpoints
    )
    logger.info("Analysis report written to %s", report_p)
    return report_p
