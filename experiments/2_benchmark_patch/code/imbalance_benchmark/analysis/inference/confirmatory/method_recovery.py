from __future__ import annotations

from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.confirmatory.holm import PRIMARY_METHODS
from imbalance_benchmark.analysis.inference.gates import (
    _SeverityInputs,
    _recovery_comparison,
)
from imbalance_benchmark.analysis.inference.permutation import (
    paired_block_permutation_ba,
    paired_block_permutation_tail_nll,
)

__all__ = ["_method_discrimination_recovery", "_method_calibration_recovery"]


def _method_discrimination_recovery(
    inp: _SeverityInputs,
    ba_deficit_dist: np.ndarray,
    severity_ba: np.ndarray,
    method: str,
    method_rec: dict[str, Any],
    gate_passed: bool,
) -> dict[str, Any]:
    """One method's discrimination-axis recovery ratio and permutation p-value."""
    method_ba = inp.ctx.ba_distribution(
        inp.balanced["labels"], method_rec["preds"], inp.n_classes
    )
    p_val = (
        paired_block_permutation_ba(
            inp.balanced["labels"],
            method_rec["preds"],
            inp.severity_ce["preds"],
            inp.ctx.case_ids,
            inp.n_classes,
            inp.n_perm,
            inp.seed,
        )
        if gate_passed and method in PRIMARY_METHODS
        else None
    )
    return _recovery_comparison(
        inp,
        method,
        "discrimination",
        method_ba - severity_ba,
        ba_deficit_dist,
        gate_passed,
        p_val,
    )


def _method_calibration_recovery(
    inp: _SeverityInputs,
    cal_deficit_dist: np.ndarray,
    severity_tail_nll: np.ndarray,
    tail_classes: list[int],
    method: str,
    method_rec: dict[str, Any],
    gate_passed: bool,
) -> dict[str, Any]:
    """One method's calibration-axis (tail NLL) recovery ratio and permutation p-value."""
    method_tail_nll = inp.ctx.tail_nll_distribution(
        inp.balanced["labels"], method_rec["probs"], tail_classes
    )
    p_val = (
        paired_block_permutation_tail_nll(
            inp.balanced["labels"],
            method_rec["probs"],
            inp.severity_ce["probs"],
            inp.ctx.case_ids,
            tail_classes,
            inp.n_perm,
            inp.seed,
        )
        if gate_passed and method in PRIMARY_METHODS
        else None
    )
    return _recovery_comparison(
        inp,
        method,
        "calibration",
        severity_tail_nll - method_tail_nll,
        cal_deficit_dist,
        gate_passed,
        p_val,
    )
