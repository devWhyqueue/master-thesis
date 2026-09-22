"""Endpoint BA-damage contrasts (H1-H3), the Spearman(S, D) check, the calibration table, and the
per-class/pair tables for exp-32's worst/flip/mild rate-weighted-score arms.

``pair_gaps``, ``confusion_partner``, and ``scores_by_key`` are score- and arm-name-agnostic and
are reused unchanged from exp-31's ``worst.report``. The rest hardcode exp-32's own arm/order names
(``worst``/``flip``/``mild`` instead of exp-31's ``sep``/``flip``/``tog``, and there is no r10 arm).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from breadth.analyze.secondary import pack_estimate

from centre.analyze import pooled

from assignment.properties import _spearman

from worst.precheck import _random_s_and_d
from worst.report import confusion_partner, pair_gaps, scores_by_key

__all__ = [
    "calibration_table",
    "confusion_partner",
    "endpoint_dists",
    "pair_gap_series",
    "pair_gaps",
    "recall_change_table",
    "scores_by_key",
    "spearman_check",
]

_TABLE_ARMS: tuple[str, ...] = (
    "worst_r100",
    "flip_r100",
    "mild_r100",
    "easy_r100",
    "hard_r100",
)


def endpoint_dists(ba: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The pre-specified BA-damage contrasts (H1-H3), from pooled arm accuracy alone."""
    dists = {f"arm_{a}": d for a, d in ba.items()}
    dists["D_worst"] = ba["r1"] - ba["worst_r100"]
    dists["D_flip"] = ba["r1"] - ba["flip_r100"]
    dists["D_mild"] = ba["r1"] - ba["mild_r100"]
    dists["D_easy"] = ba["r1"] - ba["easy_r100"]
    dists["D_hard"] = ba["r1"] - ba["hard_r100"]
    dists["D_r100_native_mean"] = ba["r1"] - ba["r100"]
    dists["h1_worst_minus_r100"] = dists["D_worst"] - dists["D_r100_native_mean"]
    dists["h2_r100_minus_mild"] = dists["D_r100_native_mean"] - dists["D_mild"]
    dists["h3_worst_minus_flip"] = dists["D_worst"] - dists["D_flip"]
    dists["mild_minus_hard"] = dists["D_mild"] - dists["D_hard"]
    return dists


def spearman_check(
    exp25_config: dict[str, Any],
    names: list[str],
    g: int,
    score_params: tuple[np.ndarray, np.ndarray, np.ndarray],
    s_by_key: dict[str, float],
    dists: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Spearman(S, D) over the 34 pre-declared orders: 30 random draws + easy/hard/worst/mild."""
    w, h, z = score_params
    s_random, d_random = _random_s_and_d(exp25_config, names, g, w, h, z)
    s_all = list(s_random) + [s_by_key[k] for k in ("easy", "hard", "worst", "mild")]
    d_all = list(d_random) + [
        float(dists[f"D_{k}"][0]) for k in ("easy", "hard", "worst", "mild")
    ]
    return {
        "spearman": _spearman(s_all, d_all),
        "s_random": s_random.tolist(),
        "d_random": d_random.tolist(),
    }


def calibration_table(
    slope: float,
    intercept: float,
    s_random: list[float],
    d_random: list[float],
    s_by_key: dict[str, float],
    dists: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Observed vs. precheck-predicted D (no refit) for the 5 fixed orders plus the random mean."""

    def _predicted(s: float) -> float:
        return intercept + slope * s

    table = {
        key: {
            "s": s_by_key[key],
            "observed": float(dists[f"D_{key}"][0]),
            "predicted": _predicted(s_by_key[key]),
        }
        for key in ("easy", "hard", "worst", "flip", "mild")
    }
    s_random_mean = float(np.mean(s_random))
    table["random_mean"] = {
        "s": s_random_mean,
        "observed": float(np.mean(d_random)),
        "predicted": _predicted(s_random_mean),
    }
    return table


def recall_change_table(
    class_acc: dict[str, np.ndarray],
    r1_own: dict[str, np.ndarray],
    weight: np.ndarray,
    names: list[str],
) -> dict[str, dict[str, Any]]:
    """Per-class own-recall change vs. r1, pooled, for the worst/flip/mild/easy/hard arms."""
    return {
        arm: {
            c: pack_estimate(pooled(class_acc[arm][:, ci, :], weight) - r1_own[c])
            for ci, c in enumerate(names)
        }
        for arm in _TABLE_ARMS
    }


def pair_gap_series(
    orders: dict[str, list[str]],
    partner: dict[str, str],
    recall_change: dict[str, dict[str, Any]],
    names: list[str],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """(gap_worst, change_worst, gap_mild, change_mild) for the pair-gap figure."""
    rank_of = {
        key: {c: i for i, c in enumerate(orders[key])} for key in ("worst", "mild")
    }
    gap_worst = [
        float(abs(rank_of["worst"][c] - rank_of["worst"][partner[c]])) for c in names
    ]
    gap_mild = [
        float(abs(rank_of["mild"][c] - rank_of["mild"][partner[c]])) for c in names
    ]
    change_worst = [recall_change["worst_r100"][c]["point"] for c in names]
    change_mild = [recall_change["mild_r100"][c]["point"] for c in names]
    return gap_worst, change_worst, gap_mild, change_mild
