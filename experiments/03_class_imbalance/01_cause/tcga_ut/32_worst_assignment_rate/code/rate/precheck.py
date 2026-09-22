"""Part A gate (0 new fits): sanity-check the rate-weighted score S against exp-30's already-
observed easy/hard/random damage, then fit and calibrate a D-on-S line to pre-register the Part B
predictions.

- G1/G2/G3 and the gate figure reuse exp-31's ``worst.precheck`` helpers unchanged: they take w, h,
  z and s_easy/s_hard/s_random/d_random as plain arrays, agnostic to how w was built.
- Calibration: an ordinary-least-squares line D = intercept + slope*S over the 30 random r100 draws
  predicts D for the held-out easy/hard anchors (compared against exp-30's observed CI) and
  pre-registers the Part B predictions for worst/flip/mild.
- Robustness: Spearman(rank, r1 recall) in the worst order -- low means the order is not just
  sorting by headroom, the failure mode that broke exp-31's row-normalized score.

Pass = G1 and G2, same as exp-31. No score tuning loop -- the score is fixed by the exp-32 plan
before this runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from breadth.analyze.canonical import canonical_class_names

from spectrum import baseline_config

from prevalence import patients_per_class

from assignment.properties import _spearman

from worst.figures import gate_figure
from worst.order import order_perm
from worst.precheck import (
    _easy_hard_scores,
    _gate_result,
    _random_s_and_d,
    gate_markers,
)
from worst.score import score

from rate.order import derive_orders, score_inputs

__all__ = ["run_precheck"]


def _fit_line(s_random: np.ndarray, d_random: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) of the OLS line D = intercept + slope*S over the random draws."""
    slope, intercept = np.polyfit(s_random, d_random, 1)
    return float(slope), float(intercept)


def _predict(slope: float, intercept: float, s: float) -> float:
    return intercept + slope * s


def _loo_spearman(s_random: np.ndarray, d_random: np.ndarray) -> float:
    """Leave-one-draw-out cross-validated Spearman between the OLS-predicted and observed D."""
    preds = []
    for i in range(len(s_random)):
        mask = np.ones(len(s_random), dtype=bool)
        mask[i] = False
        slope, intercept = _fit_line(s_random[mask], d_random[mask])
        preds.append(_predict(slope, intercept, s_random[i]))
    return _spearman(preds, list(d_random))["spearman_r"]


def _rank_vs_h(worst_order: list[str], names: list[str], h: np.ndarray) -> float:
    """Spearman(rank, r1 recall) in the worst order -- avoids exp-31's headroom-sort failure mode."""
    h_by_name = dict(zip(names, h))
    ranks = [float(i) for i in range(len(worst_order))]
    values = [float(h_by_name[c]) for c in worst_order]
    return _spearman(ranks, values)["spearman_r"]


def _pre_registered_scores(
    exp25_config: dict[str, Any],
    names: list[str],
    g: int,
    w: np.ndarray,
    h: np.ndarray,
    z: np.ndarray,
) -> tuple[
    float, float, np.ndarray, np.ndarray, dict[str, list[str]], dict[str, float]
]:
    """s_easy, s_hard, the 30 random (S, D) draws, the derived orders, and S per named order."""
    s_easy, s_hard = _easy_hard_scores(exp25_config, names, w, h, z)
    s_random, d_random = _random_s_and_d(exp25_config, names, g, w, h, z)
    orders = derive_orders(exp25_config, names, g)
    s_by_key = {
        key: score(order_perm(orders[key], names), w, h, z)
        for key in ("worst", "flip", "mild")
    }
    s_by_key["easy"], s_by_key["hard"] = s_easy, s_hard
    return s_easy, s_hard, s_random, d_random, orders, s_by_key


def _calibration(
    s_random: np.ndarray, d_random: np.ndarray, s_by_key: dict[str, float]
) -> dict[str, Any]:
    """Fitted D-on-S line, its LOO Spearman, and the pre-registered Part B predictions."""
    slope, intercept = _fit_line(s_random, d_random)
    return {
        "fit": {
            "slope": slope,
            "intercept": intercept,
            "loo_spearman": _loo_spearman(s_random, d_random),
        },
        "predicted": {k: _predict(slope, intercept, s) for k, s in s_by_key.items()},
    }


def _held_out_check(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Observed CI (from exp-30) and whether the calibration line's prediction falls inside it."""
    observed = {
        name: {"point": d_obs, "ci_2_5": lo, "ci_97_5": hi}
        for name, (_, d_obs, lo, hi) in gate_markers(config, result).items()
    }
    held_out_in_ci = {
        name: observed[name]["ci_2_5"]
        <= result["predicted"][name]
        <= observed[name]["ci_97_5"]
        for name in ("easy", "hard")
    }
    return {"observed": observed, "held_out_in_ci": held_out_in_ci}


def _write_outputs(
    config: dict[str, Any],
    result: dict[str, Any],
    s_random: np.ndarray,
    d_random: np.ndarray,
) -> None:
    write_json(output_root(config) / "data" / "precheck.json", result)
    gate_figure(
        s_random,
        d_random,
        gate_markers(config, result),
        output_root(config) / "figures" / "gate_score_vs_damage.pdf",
    )


def run_precheck(config: dict[str, Any]) -> dict[str, Any]:
    """Compute G1/G2/G3, fit and calibrate the D-on-S line, and write ``precheck.json``."""
    names = canonical_class_names(config)
    exp25_config = baseline_config(config, "prevalence_outputs")
    g = patients_per_class(exp25_config)
    w, h, z = score_inputs(exp25_config, names, g)

    s_easy, s_hard, s_random, d_random, orders, s_by_key = _pre_registered_scores(
        exp25_config, names, g, w, h, z
    )
    result = _gate_result(s_easy, s_hard, s_random, d_random)

    calibration = _calibration(s_random, d_random, s_by_key)
    result["fit"] = {
        **calibration["fit"],
        "spearman": result["g3_spearman"]["spearman_r"],
    }
    result["predicted"] = calibration["predicted"]
    result.update(_held_out_check(config, result))
    result["rank_vs_h"] = _rank_vs_h(orders["worst"], names, h)

    _write_outputs(config, result, s_random, d_random)
    return result
