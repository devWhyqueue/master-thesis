"""Gate 4: train-time logit adjustment equals post-hoc adjustment of CE (shard 0/0, r100, tau 1).

Both arms are fit by the same solver at the same precision (``fit_offset_logistic``; post-hoc =
zero offset, then logits - log pi), so the gate tests the identity, not solver tolerance. It
was re-specified on 2026-11-09 after the first version, which paired exp-25's sklearn recipe
(tol 1e-8, stops far from the optimum) with a tight offset fit, failed on both datasets. That
recipe's distance to the converged CE is reported next to the gate as information (issue #5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from decodability.linear import fit_multinomial_logistic
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _macro_recall_mean,
)
from imbalance_benchmark.common import read_run_record

from breadth import MAX_ITER, TOLERANCE
from breadth.fit import EvalPartition, init_shard

from prevalence import patients_per_class
from prevalence.fit import _arm_rows, _shard_context, class_counts

from intended import damage_dir
from intended.lam import OffsetFit, fit_offset_logistic, logits

__all__ = ["la_equivalence"]

TIE_GAP = 1e-6
MAX_BA_DIFF_PP = 0.01
ISSUE = "https://github.com/devWhyqueue/master-thesis/issues/5"
_MISSING = "stored exp-25/26 r100 record of shard (0, 0) missing"


def _compare(a: np.ndarray, b: np.ndarray, evals: EvalPartition) -> dict[str, Any]:
    """Prediction disagreement outside near-ties of ``a`` and the patient-macro BA difference."""
    top2 = np.sort(a, axis=1)[:, -2:]
    clear = top2[:, 1] - top2[:, 0] > TIE_GAP
    cases = evals.test_id["case_id"].astype(str).to_numpy()
    ba = [_macro_recall_mean(evals.test_y, z.argmax(1), cases) for z in (a, b)]
    return {
        "differing_rows": int((a.argmax(1) != b.argmax(1)).sum()),
        "differing_clear_rows": int((a.argmax(1) != b.argmax(1))[clear].sum()),
        "near_tie_rows": int((~clear).sum()),
        "ba_difference_pp": abs(ba[0] - ba[1]) * 100,
    }


def _solver(fit: OffsetFit) -> dict[str, Any]:
    return {
        "converged": fit.converged,
        "gradient_max": fit.gradient_max,
        "iterations": fit.iterations,
    }


def _shard_rows(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EvalPartition]:
    """Training rows, targets and class counts of shard (0, 0) at r100, plus its eval partitions."""
    train_df, names, evals, _ = init_shard(config, 0)
    shard = _shard_context(train_df, names, 0, 0, patients_per_class(config))
    counts = class_counts(
        "r100", shard.perm, shard.available, shard.pool_counts, shard.g
    )
    x, y = _arm_rows(train_df, names, shard.patients, counts)
    return x, y, np.asarray(counts), evals


def _passes(fits: tuple[OffsetFit, OffsetFit], gate: dict[str, Any]) -> bool:
    converged = all(fit.converged for fit in fits)
    agree = gate["differing_clear_rows"] == 0
    return converged and agree and gate["ba_difference_pp"] < MAX_BA_DIFF_PP


def la_equivalence(config: dict[str, Any]) -> dict[str, Any]:
    """Gate-4 result: passes iff both fits converged, clear rows agree, BA differs < 0.01 pp."""
    x, y, counts, evals = _shard_rows(config)
    stored = read_run_record(
        damage_dir(config, 0, "r100", 0), splits=(), array_fields=()
    )
    if stored is None:
        return {"passed": False, "problems": [_MISSING]}
    lam, log_pi = float(stored["selected_lambda"]), np.log(counts / counts.sum())
    test_x = np.asarray(evals.test_x, dtype=np.float64)
    plain, trained = (
        fit_offset_logistic(x, y, lam, o) for o in (np.zeros_like(log_pi), log_pi)
    )
    ce = test_x @ plain.coef.T + plain.intercept
    gate = _compare(ce - log_pi, test_x @ trained.coef.T + trained.intercept, evals)
    recipe = logits(fit_multinomial_logistic(x, y, lam, TOLERANCE, MAX_ITER), test_x)
    return {
        "passed": _passes((plain, trained), gate),
        "lambda": lam,
        "tau": 1.0,
        **gate,
        "solvers": {"posthoc": _solver(plain), "trained": _solver(trained)},
        "exp25_recipe_vs_converged_ce": {**_compare(ce, recipe, evals), "issue": ISSUE},
    }
