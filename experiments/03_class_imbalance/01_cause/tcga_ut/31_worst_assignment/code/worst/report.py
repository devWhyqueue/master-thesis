"""Endpoint BA-damage contrasts, the Spearman(S, D) check, and the per-class/pair tables."""

from __future__ import annotations

from typing import Any

import numpy as np
from breadth.analyze.secondary import pack_estimate

from centre.analyze import pooled

from assignment.properties import _confusion_counts, _spearman

from worst import CONFUSION_PAIRS
from worst.order import order_perm
from worst.precheck import _random_s_and_d
from worst.score import score

__all__ = [
    "confusion_partner",
    "endpoint_dists",
    "pair_gap_series",
    "pair_gaps",
    "recall_change_table",
    "scores_by_key",
    "spearman_check",
]

_TABLE_ARMS: tuple[str, ...] = (
    "sep_r100",
    "flip_r100",
    "tog_r100",
    "easy_r100",
    "hard_r100",
)


def endpoint_dists(ba: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The pre-specified BA-damage contrasts, from pooled arm accuracy alone."""
    dists = {f"arm_{a}": d for a, d in ba.items()}
    dists["D_sep"] = ba["r1"] - ba["sep_r100"]
    dists["D_flip"] = ba["r1"] - ba["flip_r100"]
    dists["D_tog"] = ba["r1"] - ba["tog_r100"]
    dists["D_easy"] = ba["r1"] - ba["easy_r100"]
    dists["D_hard"] = ba["r1"] - ba["hard_r100"]
    dists["D_r100_native_mean"] = ba["r1"] - ba["r100"]
    dists["D_r10_native_mean"] = ba["r1"] - ba["r10"]
    dists["D_sep_r10"] = ba["r1"] - ba["sep_r10"]
    dists["sep_minus_r100"] = dists["D_sep"] - dists["D_r100_native_mean"]
    dists["sep_minus_tog"] = dists["D_sep"] - dists["D_tog"]
    dists["sep_minus_flip"] = dists["D_sep"] - dists["D_flip"]
    dists["tog_minus_r100"] = dists["D_tog"] - dists["D_r100_native_mean"]
    dists["sep_r10_minus_r10"] = dists["D_sep_r10"] - dists["D_r10_native_mean"]
    return dists


def scores_by_key(
    orders: dict[str, list[str]],
    names: list[str],
    w: np.ndarray,
    h: np.ndarray,
    z: np.ndarray,
) -> dict[str, float]:
    """S(pi) for every named order (sep/flip/tog/easy/hard), canonical class-index space."""
    return {
        key: score(order_perm(rank_order, names), w, h, z)
        for key, rank_order in orders.items()
    }


def spearman_check(
    exp25_config: dict[str, Any],
    names: list[str],
    g: int,
    score_params: tuple[np.ndarray, np.ndarray, np.ndarray],
    s_by_key: dict[str, float],
    dists: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Spearman(S, D) over the 34 pre-declared orders: 30 random draws + easy/hard/sep/tog."""
    w, h, z = score_params
    s_random, d_random = _random_s_and_d(exp25_config, names, g, w, h, z)
    s_all = list(s_random) + [s_by_key[k] for k in ("easy", "hard", "sep", "tog")]
    d_all = list(d_random) + [
        float(dists[f"D_{k}"][0]) for k in ("easy", "hard", "sep", "tog")
    ]
    return {
        "spearman": _spearman(s_all, d_all),
        "s_random": s_random.tolist(),
        "d_random": d_random.tolist(),
    }


def confusion_partner(exp25_config: dict[str, Any], names: list[str]) -> dict[str, str]:
    """Each class's single strongest confusion partner (symmetric confusion mass, off-diagonal)."""
    mat = _confusion_counts(exp25_config, names).astype(np.float64)
    sym = mat + mat.T
    np.fill_diagonal(sym, -1.0)
    return {c: names[int(np.argmax(sym[ci]))] for ci, c in enumerate(names)}


def pair_gaps(orders: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Rank gap between each named confusion pair, per order."""
    out: dict[str, dict[str, int]] = {}
    for c, d in CONFUSION_PAIRS:
        gaps = {
            key: abs(rank_order.index(c) - rank_order.index(d))
            for key, rank_order in orders.items()
            if c in rank_order and d in rank_order
        }
        out[f"{c}|{d}"] = gaps
    return out


def recall_change_table(
    class_acc: dict[str, np.ndarray],
    r1_own: dict[str, np.ndarray],
    weight: np.ndarray,
    names: list[str],
) -> dict[str, dict[str, Any]]:
    """Per-class own-recall change vs. r1, pooled, for the sep/flip/tog/easy/hard arms."""
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
    """(gap_sep, change_sep, gap_tog, change_tog) for the pair-gap figure."""
    rank_of = {key: {c: i for i, c in enumerate(orders[key])} for key in ("sep", "tog")}
    gap_sep = [
        float(abs(rank_of["sep"][c] - rank_of["sep"][partner[c]])) for c in names
    ]
    gap_tog = [
        float(abs(rank_of["tog"][c] - rank_of["tog"][partner[c]])) for c in names
    ]
    change_sep = [recall_change["sep_r100"][c]["point"] for c in names]
    change_tog = [recall_change["tog_r100"][c]["point"] for c in names]
    return gap_sep, change_sep, gap_tog, change_tog
