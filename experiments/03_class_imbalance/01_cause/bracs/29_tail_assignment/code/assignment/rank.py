"""Per-rank bookkeeping: shift/rank derivation, and one ratio's tail-damage bundle.

Every fit stores its own realized ``class_counts`` (assignment/fit.py), so a fit's rank
assignment is read back from that record rather than recomputed from its shift index.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from centre import N_DRAWS, N_SPLITS
from centre.analyze import pooled

from sites import allocation_dir

from assignment._io import paths_by_split, require_record

__all__ = ["Pooled", "TailDamage", "tail_damage_for_rho"]


def _arm_name(k: int, rho: int) -> str:
    """Fit-array arm name for shift ``k`` (0 = exp-26's own permutation, reused) at ``rho``."""
    return f"r{rho}" if k == 0 else f"a{k}_r{rho}"


def _rank_of(counts: dict[str, float], names: list[str]) -> dict[str, int]:
    """Assigned rank per class (0 = head/largest count, ``len(names) - 1`` = tail/smallest)."""
    order = sorted(names, key=lambda n: -counts[n])
    return {name: rank for rank, name in enumerate(order)}


def _rank_grid(
    config: dict[str, Any], exp26_config: dict[str, Any], names: list[str], rho: int
) -> dict[tuple[int, int, int], dict[str, int]]:
    """``rank_of[(split, draw, shift)]``, shift 0..6, read from each fit's stored class_counts."""
    paths_new, paths_reused = paths_by_split(config), paths_by_split(exp26_config)
    out: dict[tuple[int, int, int], dict[str, int]] = {}
    for s in range(N_SPLITS):
        for d in range(N_DRAWS):
            for k in range(len(names)):
                paths = paths_reused if k == 0 else paths_new
                rec = require_record(allocation_dir(paths[s], _arm_name(k, rho), d))
                out[(s, d, k)] = _rank_of(rec["class_counts"], names)
    return out


def _select_arms(
    rank_grid: dict[tuple[int, int, int], dict[str, int]], names: list[str], rho: int
) -> dict[tuple[str, int], dict[tuple[int, int], str]]:
    """``(class, rank) -> {(split, draw): arm}``: the one shift per draw realizing that rank."""
    out: dict[tuple[str, int], dict[tuple[int, int], str]] = {
        (c, r): {} for c in names for r in range(len(names))
    }
    for (s, d, k), ranks in rank_grid.items():
        arm = _arm_name(k, rho)
        for c in names:
            out[(c, ranks[c])][(s, d)] = arm
    return out


def _stack(
    class_acc: dict[str, np.ndarray],
    arm_map: dict[tuple[int, int], str],
    class_idx: int | None,
) -> np.ndarray:
    """(F, R): macro-BA (``class_idx=None``) or one class's own recall, one row per (split, draw)."""
    rows = []
    for s in range(N_SPLITS):
        for d in range(N_DRAWS):
            block = class_acc[arm_map[(s, d)]][s * N_DRAWS + d]  # (C, R)
            rows.append(block.mean(axis=0) if class_idx is None else block[class_idx])
    return np.stack(rows)


def _own_rank_r2(delta: np.ndarray, class_idx: np.ndarray, rank: np.ndarray) -> float:
    """Variance of per-(class, fit) own-recall change explained by (class, rank) group means."""
    group = class_idx * 1000 + rank
    means = {g: float(delta[group == g].mean()) for g in np.unique(group)}
    predicted = np.array([means[g] for g in group])
    ss_res = float(np.sum((delta - predicted) ** 2))
    ss_tot = float(np.sum((delta - delta.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _flat_deltas(
    class_acc: dict[str, np.ndarray],
    rank_grid: dict[tuple[int, int, int], dict[str, int]],
    names: list[str],
    rho: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (class, fit) point-estimate own-recall change vs. r1, with its class index and rank."""
    delta, class_idx, rank = [], [], []
    for (s, d, k), ranks in rank_grid.items():
        arm, i = _arm_name(k, rho), s * N_DRAWS + d
        for ci, c in enumerate(names):
            delta.append(class_acc[arm][i, ci, 0] - class_acc["r1"][i, ci, 0])
            class_idx.append(ci)
            rank.append(ranks[c])
    return np.array(delta), np.array(class_idx), np.array(rank)


def _class_tail_row(
    class_acc: dict[str, np.ndarray],
    arm_for: dict[tuple[str, int], dict[tuple[int, int], str]],
    names: list[str],
    ci: int,
    c: str,
    ba_r1: np.ndarray,
    r1_dist: np.ndarray,
    w: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One class's own-delta row over every rank, plus its tail D/own-loss/net-other."""
    n = len(names)
    row = np.zeros(n)
    tail: dict[str, np.ndarray] = {}
    for rank in range(n):
        arm_map = arm_for[(c, rank)]
        own_dist = pooled(_stack(class_acc, arm_map, ci), w) - r1_dist
        row[rank] = own_dist[0]
        if rank == n - 1:
            ba_dist = ba_r1 - pooled(_stack(class_acc, arm_map, None), w)
            own_loss_dist = own_dist / float(n)
            tail = {
                "D": ba_dist,
                "own_loss": own_loss_dist,
                "net_other": ba_dist - own_loss_dist,
            }
    return tail, row


class Pooled(NamedTuple):
    """Every arm's (F, C, R) per-class accuracy, pooled macro-BA, r1's own recall, and the draw weights."""

    class_acc: dict[str, np.ndarray]
    ba: dict[str, np.ndarray]
    r1_own: dict[str, np.ndarray]
    w: np.ndarray


class TailDamage(NamedTuple):
    """One ratio's tail-assignment bundle: distributions, the rank matrix, and the R^2 check."""

    dists: dict[str, np.ndarray]
    matrix: np.ndarray
    r2: float
    d_by_class: dict[str, np.ndarray]
    own_loss_by_class: dict[str, np.ndarray]


def tail_damage_for_rho(
    config: dict[str, Any],
    exp26_config: dict[str, Any],
    names: list[str],
    pooled_acc: Pooled,
    rho: int,
    atypical: tuple[str, ...],
) -> TailDamage:
    """Every class's tail damage, own-loss split, and rank-response row, at one ratio."""
    class_acc, ba, r1_own, w = pooled_acc
    rank_grid = _rank_grid(config, exp26_config, names, rho)
    arm_for = _select_arms(rank_grid, names, rho)
    matrix = np.zeros((len(names), len(names)))
    dists: dict[str, np.ndarray] = {}
    d_by_class: dict[str, np.ndarray] = {}
    own_loss_by_class: dict[str, np.ndarray] = {}
    for ci, c in enumerate(names):
        tail, matrix[ci] = _class_tail_row(
            class_acc, arm_for, names, ci, c, ba["r1"], r1_own[c], w
        )
        dists[f"D_{rho}_{c}"] = d_by_class[c] = tail["D"]
        dists[f"own_loss_{rho}_{c}"] = own_loss_by_class[c] = tail["own_loss"]
        dists[f"net_other_{rho}_{c}"] = tail["net_other"]
    dists[f"mean_D_{rho}"] = np.mean(np.stack(list(d_by_class.values())), axis=0)
    rest = [d_by_class[c] for c in names if c not in atypical]
    atyp = [d_by_class[c] for c in atypical]
    dists[f"atypical_minus_rest_{rho}"] = np.mean(np.stack(atyp), axis=0) - np.mean(
        np.stack(rest), axis=0
    )
    delta, class_idx, rank = _flat_deltas(class_acc, rank_grid, names, rho)
    r2 = _own_rank_r2(delta, class_idx, rank)
    return TailDamage(dists, matrix, r2, d_by_class, own_loss_by_class)
