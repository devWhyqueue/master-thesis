"""Analyze stage: BA per arm (reused r/N from exp-25, new P/S/LP/Lr from this experiment),
prior-vs-support contrasts, acceleration, and diagnostics.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names
from breadth.calibrate import read_temperature

from sites import allocation_dir

from decomposition.model import draw_weights

from centre import N_DRAWS, N_SPLITS
from centre.analyze import arm_accuracy, pooled

from spectrum import baseline_config

from directions.analyze import _write_analysis

from prevalence import RATIOS
from prevalence.analyze import _paths, _realized_rho, _require_record, _slope, _thirds

from cause import ADJUSTED_ARMS, NEW_FIT_ARMS, P_ARMS, RATIOS_NEW, REUSED_ARMS, S_ARMS
from cause.figures import component_figure

__all__ = ["run_analyze"]


def _lambda_temperature(
    config: dict[str, Any], arms: tuple[str, ...]
) -> dict[str, Any]:
    """Selected-lambda counts and mean validation temperature per arm, over every fit."""
    paths = _paths(config)
    out: dict[str, Any] = {}
    for arm in arms:
        lambdas: list[float] = []
        temps: list[float] = []
        for s in range(N_SPLITS):
            for d in range(N_DRAWS):
                result_dir = allocation_dir(paths[s], arm, d)
                lambdas.append(float(_require_record(result_dir)["selected_lambda"]))
                temps.append(read_temperature(result_dir))
        out[arm] = {
            "lambda_counts": {str(k): v for k, v in Counter(lambdas).items()},
            "mean_temperature": float(np.mean(temps)),
        }
    return out


def _combine(ba: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """r-vs-P-vs-S contrasts, interaction, shares, and acceleration per family, per rho."""
    dists = {f"arm_{arm}": d for arm, d in ba.items()}
    for r in RATIOS_NEW:
        d_r, d_p, d_s = (ba[f"{f}{r}"] - ba["r1"] for f in ("r", "P", "S"))
        dists[f"delta_r_{r}"], dists[f"delta_P_{r}"], dists[f"delta_S_{r}"] = (
            d_r,
            d_p,
            d_s,
        )
        dists[f"interaction_{r}"] = d_r - d_p - d_s
        dists[f"share_P_{r}"] = d_p / d_r
        dists[f"share_S_{r}"] = d_s / d_r
        dists[f"intercept_share_P_{r}"] = (ba[f"LP{r}"] - ba[f"P{r}"]) / d_p
        dists[f"Lr_minus_S_{r}"] = ba[f"Lr{r}"] - ba[f"S{r}"]
    dists["slope_r_le10"] = _slope(ba, RATIOS[:4], family="r")
    dists["slope_r_ge10"] = _slope(ba, RATIOS[3:], family="r")
    dists["accel_r"] = dists["slope_r_ge10"] - dists["slope_r_le10"]
    for fam in ("P", "S"):
        lo = _slope(ba, RATIOS_NEW[:3], family=fam)
        hi = _slope(ba, RATIOS_NEW[2:], family=fam)
        dists[f"slope_{fam}_le10"], dists[f"slope_{fam}_ge10"] = lo, hi
        dists[f"accel_{fam}"] = hi - lo
    return dists


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool reused (exp-25) and new (P/S/LP/Lr) arm accuracy, write analysis, diagnostics, figure."""
    names = canonical_class_names(config)
    exp25_config = baseline_config(config, "prevalence_outputs")
    acc = {
        **arm_accuracy(exp25_config, names, arms=REUSED_ARMS),
        **arm_accuracy(config, names, arms=NEW_FIT_ARMS + ADJUSTED_ARMS),
    }
    n_replicates = next(iter(acc.values())).shape[-1]
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split, N_DRAWS, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    ba = {arm: pooled(a, w) for arm, a in acc.items()}
    dists = _combine(ba)
    path = _write_analysis(config, acc, fit_split, dists)
    # r/S: realized rho of the data actually drawn. P's own data is r1's balanced draw (rho ~ 1
    # by construction), so its meaningful rho is the *prior* it was reweighted toward, which
    # equals r{rho}'s realized rho; the figure's shared x-axis uses that value for all three.
    rho_r = _realized_rho(exp25_config, REUSED_ARMS)
    rho_new = _realized_rho(config, P_ARMS + S_ARMS)
    write_json(
        path.with_name("diagnostics.json"),
        {
            "realized_rho": {**rho_r, **rho_new},
            "rank_recall": _thirds(config, P_ARMS + S_ARMS, names),
            "lambda_temperature": _lambda_temperature(config, P_ARMS + S_ARMS),
        },
    )
    component_figure(dists, rho_r, output_root(config) / "figures" / "component.pdf")
    return path
