"""Cohort search: batched best-swap descent on coverage, hull residual, and similarity targets."""

from __future__ import annotations

import numpy as np

from hull import MAX_SWAP_PASSES, N_STARTS
from hull.design import Target
from hull.geometry import HullGeometry, pool_values

__all__ = ["loss", "swap_search", "search_cohort"]


def loss(geo: HullGeometry, idx: np.ndarray, target: Target) -> np.ndarray:
    """(N,) weighted distance of (N, G) cohorts to a (r, h, omega) target on the pool."""
    r, h, omega = pool_values(geo, idx)
    return (
        np.abs(r - target.r) / target.tol_r
        + np.abs(h - target.h) / target.tol_h
        + np.abs(omega - target.omega) / target.tol_omega
    )


def _best_swap(
    geo: HullGeometry, idx: np.ndarray, target: Target, current: float
) -> np.ndarray | None:
    """The single (out, in) replacement that lowers the loss most, or None."""
    outside = np.setdiff1d(np.arange(len(geo.base.pool)), idx)
    best_loss, best_idx = current, None
    for k in range(len(idx)):
        rest = np.delete(idx, k)
        batch = np.column_stack([np.repeat(rest[None, :], len(outside), 0), outside])
        losses = loss(geo, batch, target)
        j = int(np.argmin(losses))
        if losses[j] < best_loss - 1e-12:
            best_loss, best_idx = float(losses[j]), batch[j]
    return best_idx


def swap_search(geo: HullGeometry, start: np.ndarray, target: Target) -> np.ndarray:
    """Repeated best-swap passes from ``start`` until no swap improves the loss."""
    idx = np.asarray(start)
    current = float(loss(geo, idx[None, :], target)[0])
    for _ in range(MAX_SWAP_PASSES):
        new_idx = _best_swap(geo, idx, target, current)
        if new_idx is None:
            break
        idx = new_idx
        current = float(loss(geo, idx[None, :], target)[0])
    return idx


def search_cohort(
    geo: HullGeometry, candidates: list[np.ndarray], target: Target
) -> np.ndarray:
    """Best swap-searched cohort, started from the ``N_STARTS`` lowest-loss candidates."""
    stacked = np.stack(candidates)
    ranked = np.argsort(loss(geo, stacked, target), kind="stable")[:N_STARTS]
    results = [swap_search(geo, stacked[i], target) for i in ranked]
    final = loss(geo, np.stack(results), target)
    return results[int(np.argmin(final))]
