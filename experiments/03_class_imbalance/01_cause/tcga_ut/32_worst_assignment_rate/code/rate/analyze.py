"""Analyze stage: pool exp-25/exp-30 reused arms with exp-32's worst/flip/mild arms, compute the
H1-H3 BA-damage contrasts, the Spearman(S, D) check, the calibration table (against precheck's
fitted line, no refit), the per-class recall-change table, and confusion-pair rank gaps. Writes
analysis.json, diagnostics.json, and two figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from breadth.analyze.canonical import canonical_class_names

from spectrum import baseline_config

from prevalence import patients_per_class

from directions.analyze import _write_analysis

from assignment.analyze import _class_recall_stack, _pool
from assignment.properties import class_properties

from permutation.order import tail_order

from worst.figures import damage_vs_score_figure, pair_gap_figure

from rate import EXP30_ARMS, NEW_FIT_ARMS, REUSED_ARMS
from rate import report
from rate.order import derive_orders, score_inputs

__all__ = ["run_analyze"]


def _prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Pool every arm's per-class accuracy and reproduce the fit stage's derived orders."""
    names = canonical_class_names(config)
    exp25_config = baseline_config(config, "prevalence_outputs")
    exp30_config = baseline_config(config, "tail_outputs")
    g = patients_per_class(exp25_config)
    class_acc = {
        **_class_recall_stack(exp25_config, names, REUSED_ARMS),
        **_class_recall_stack(exp30_config, names, EXP30_ARMS),
        **_class_recall_stack(config, names, NEW_FIT_ARMS),
    }
    acc, fit_split, weight, ba, r1_own = _pool(class_acc, names)
    tail = tail_order(exp25_config, names)
    orders = {
        **derive_orders(exp25_config, names, g),
        "easy": tail,
        "hard": list(reversed(tail)),
    }
    w, h, z = score_inputs(exp25_config, names, g)
    precheck = json.loads(
        (output_root(config) / "data" / "precheck.json").read_text(encoding="utf-8")
    )
    return {
        "names": names,
        "exp25_config": exp25_config,
        "g": g,
        "class_acc": class_acc,
        "acc": acc,
        "fit_split": fit_split,
        "weight": weight,
        "ba": ba,
        "r1_own": r1_own,
        "orders": orders,
        "w": w,
        "h": h,
        "z": z,
        "precheck": precheck,
    }


def _write_diagnostics(path: Path, p: dict[str, Any], r: dict[str, Any]) -> None:
    write_json(
        path.with_name("diagnostics.json"),
        {
            "orders": p["orders"],
            "score": r["s_by_key"],
            "spearman_score_vs_damage": r["spearman"]["spearman"],
            "calibration": r["calibration"],
            "random_orders": {
                "s": r["spearman"]["s_random"],
                "d": r["spearman"]["d_random"],
            },
            "class_properties": r["properties"],
            "confusion_partner": r["partner"],
            "confusion_pair_gaps": r["gaps"],
            "per_class_recall_change": r["recall_change"],
        },
    )


def _write_figures(
    config: dict[str, Any], p: dict[str, Any], r: dict[str, Any]
) -> None:
    figures = output_root(config) / "figures"
    points = {
        key: (r["s_by_key"][key], r["dists"][f"D_{key}"])
        for key in ("easy", "hard", "worst", "flip", "mild")
    }
    damage_vs_score_figure(
        np.array(r["spearman"]["s_random"]),
        np.array(r["spearman"]["d_random"]),
        points,
        figures / "damage_vs_score_r100.pdf",
    )
    gap_worst, change_worst, gap_mild, change_mild = report.pair_gap_series(
        p["orders"], r["partner"], r["recall_change"], p["names"]
    )
    pair_gap_figure(
        gap_worst, change_worst, gap_mild, change_mild, figures / "pair_gap_r100.pdf"
    )


def _compute(p: dict[str, Any]) -> dict[str, Any]:
    """Every endpoint, check, calibration table, and per-class/pair table the report needs."""
    dists = report.endpoint_dists(p["ba"])
    s_by_key = report.scores_by_key(p["orders"], p["names"], p["w"], p["h"], p["z"])
    spearman = report.spearman_check(
        p["exp25_config"], p["names"], p["g"], (p["w"], p["h"], p["z"]), s_by_key, dists
    )
    fit = p["precheck"]["fit"]
    return {
        "dists": dists,
        "s_by_key": s_by_key,
        "spearman": spearman,
        "calibration": report.calibration_table(
            fit["slope"],
            fit["intercept"],
            spearman["s_random"],
            spearman["d_random"],
            s_by_key,
            dists,
        ),
        "properties": class_properties(p["exp25_config"], p["names"], p["r1_own"]),
        "partner": report.confusion_partner(p["exp25_config"], p["names"]),
        "gaps": report.pair_gaps(p["orders"]),
        "recall_change": report.recall_change_table(
            p["class_acc"], p["r1_own"], p["weight"], p["names"]
        ),
    }


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool reused and new-order arm accuracy; compute contrasts, checks, and write the report."""
    p = _prepare(config)
    r = _compute(p)
    path = _write_analysis(config, p["acc"], p["fit_split"], r["dists"])
    _write_diagnostics(path, p, r)
    _write_figures(config, p, r)
    return path
