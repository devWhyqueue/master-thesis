"""Run-record stage: score frozen winners on validation and test and write per-arm run records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from decodability.linear import predict_logreg
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _cluster_discrimination,
    clustered_endpoints,
)
from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    read_run_record,
    write_run_record,
)

from breadth.fit import EvalPartition, _build_draw_record

from sites import allocation_dir

from centre import patches_per_patient

from uncertainty import MAX_ITER, TOLERANCE, baseline_arm_dir
from uncertainty.freeze import WINNERS_NAME
from uncertainty.loss import UncertainFit, as_linear_result

__all__ = ["write_arm_records"]


def _evaluate(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    meta: tuple[int, int, int],
    lam: float,
    fit: UncertainFit,
    evals: EvalPartition,
    extra: dict[str, Any],
) -> None:
    """Score a frozen winner on validation and test and write its run record."""
    g, n, draw_idx = meta
    result = as_linear_result(fit, lam, n, TOLERANCE, MAX_ITER)
    cases = evals.val_id["case_id"].astype(str).to_numpy()
    slides = evals.val_id["slide_id"].astype(str).to_numpy()
    v_preds, _ = predict_logreg(evals.val_x, fit.coef, fit.intercept)
    val_end = _cluster_discrimination(evals.val_y, v_preds, cases, slides, is_mil=False)
    preds, probs = predict_logreg(evals.test_x, fit.coef, fit.intercept)
    test_end = clustered_endpoints(
        evals.test_y, preds, probs, evals.test_id, is_mil=False
    )
    rec = _build_draw_record(
        config,
        (g, patches_per_patient(g), draw_idx),
        lam,
        result,
        (preds, probs, val_end, test_end),
        evals.test_y,
    )
    write_run_record(out_dir, {**rec, "arm": arm, **extra}, keep_arrays=True)


def _copy_baseline(
    config: dict[str, Any], out_dir: Path, arm: str, where: tuple[int, int, int]
) -> None:
    """Write the baseline R record of this shard as ``arm``'s record (t = 0 won selection)."""
    split_idx, draw_idx, g = where
    base = read_run_record(baseline_arm_dir(config, split_idx, draw_idx, f"R{g}"))
    assert base is not None
    extra = {"arm": arm, "t": 0.0, "covariance": None}
    write_run_record(out_dir, {**base, **extra}, keep_arrays=True)


def write_arm_records(
    config: dict[str, Any],
    paths: dict[str, Path],
    record: dict[str, Any],
    sel_dir: Path,
    evals: EvalPartition,
    n: int,
    where: tuple[int, int, int],
) -> None:
    """Write every missing arm run record of one patient count from the frozen selection."""
    _, draw_idx, g = where
    with np.load(sel_dir / WINNERS_NAME) as npz:
        winners = {k: npz[k] for k in npz.files}
    for family, sel in record["selected"].items():
        arm = f"{family}{g}"
        out_dir = allocation_dir(paths, arm, draw_idx)
        if (out_dir / RUN_RECORD_NAME).exists():
            continue
        if sel is None:  # t = 0 wins: R's record is this arm's record.
            _copy_baseline(config, out_dir, arm, where)
            continue
        fit = UncertainFit(
            winners[f"{family}_coef"], winners[f"{family}_intercept"], 0.0, 0.0, 0, True
        )
        extra = {
            "t": sel["t"],
            "covariance": sel["kind"],
            "selection_fingerprint": record["fingerprint"],
        }
        _evaluate(config, out_dir, arm, (g, n, draw_idx), sel["lam"], fit, evals, extra)
