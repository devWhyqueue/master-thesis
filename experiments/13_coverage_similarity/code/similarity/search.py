"""Cohort construction: candidate generation and swap search (report App. A)."""

from __future__ import annotations

from functools import partial
from typing import Callable, Iterator, NamedTuple

import numpy as np

from breadth.sampling import derive_draw_seed

from similarity import (
    MAX_SWAP_PASSES,
    N_STARTS,
    NEFF_REL_TOL,
    OMEGA_TOL,
    R_TOL,
    SEARCH_BASE_SEED,
    TARGET_OMEGA_GAP,
    TARGET_R_GAP,
)
from similarity.candidates import candidates as _build_candidates
from similarity.geometry import ClassGeometry, neff_of, omega_of, r_of

__all__ = ["CohortResult", "search_class_draw", "search_split_draw"]

LossFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class CohortResult(NamedTuple):
    """One cohort's selected patients and its training/validation quantities."""

    patients: list[str]
    r_train: float
    r_val: float
    omega: float
    neff: float | None


def _incremental_candidates(
    idx: np.ndarray, geo: ClassGeometry
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Per swap-out position k, the vectorized (r, omega) of every swap-in patient."""
    g = len(idx)
    idx_set = set(idx.tolist())
    candidates_in = np.array([p for p in range(len(geo.pool)) if p not in idx_set])
    sub = geo.gram[np.ix_(idx, idx)]
    offdiag_s = float(sub.sum() - np.trace(sub))
    denom_omega = g * (g - 1) * geo.tau2 if geo.tau2 > 0 else None
    for k in range(g):
        rest = np.delete(idx, k)
        min_excl = (
            geo.d_pool[:, rest].min(axis=1)
            if g > 1
            else np.full(geo.d_pool.shape[0], np.inf)
        )
        r_cand = np.minimum(min_excl[:, None], geo.d_pool[:, candidates_in]).mean(
            axis=0
        )
        if denom_omega is not None:
            removed_sum = float(geo.gram[idx[k], rest].sum()) if g > 1 else 0.0
            cross = (
                geo.gram[np.ix_(candidates_in, rest)].sum(axis=1)
                if g > 1
                else np.zeros(len(candidates_in))
            )
            omega_cand = (offdiag_s - 2.0 * removed_sum + 2.0 * cross) / denom_omega
        else:
            omega_cand = np.zeros(len(candidates_in))
        yield k, candidates_in, r_cand, omega_cand


def _best_swap(
    idx: np.ndarray, geo: ClassGeometry, loss_fn: LossFn
) -> np.ndarray | None:
    """One swap pass: the single best (out, in) replacement under ``loss_fn``, or None."""
    pool_ids = np.asarray(geo.pool)
    cur_loss = float(
        loss_fn(np.array([r_of(geo.d_pool, idx)]), np.array([omega_of(geo, idx)]))[0]
    )
    best_loss, best_swap = cur_loss, None
    for k, candidates_in, r_cand, omega_cand in _incremental_candidates(idx, geo):
        loss_cand = loss_fn(r_cand, omega_cand)
        order = np.lexsort((pool_ids[candidates_in], loss_cand))
        j_local = int(order[0])
        loss_val = float(loss_cand[j_local])
        j = int(candidates_in[j_local])
        if loss_val < best_loss or (
            loss_val == best_loss
            and best_swap is not None
            and pool_ids[j] < pool_ids[best_swap[1]]
        ):
            best_loss, best_swap = loss_val, (k, j)
    if best_swap is None:
        return None
    k, j = best_swap
    new_idx = idx.copy()
    new_idx[k] = j
    return new_idx


def _swap_search(
    start_idx: np.ndarray, geo: ClassGeometry, loss_fn: LossFn
) -> np.ndarray:
    """Repeated best-swap passes; ties break toward the smaller patient identifier."""
    idx = start_idx.copy()
    for _ in range(MAX_SWAP_PASSES):
        new_idx = _best_swap(idx, geo, loss_fn)
        if new_idx is None:
            break
        idx = new_idx
    return idx


def _score(idx: np.ndarray, geo: ClassGeometry, loss_fn: LossFn) -> float:
    return float(
        loss_fn(np.array([r_of(geo.d_pool, idx)]), np.array([omega_of(geo, idx)]))[0]
    )


def _search(
    candidates: list[np.ndarray], geo: ClassGeometry, loss_fn: LossFn
) -> np.ndarray:
    """Best swap-searched cohort, started from the top ``N_STARTS`` candidates."""
    ranked = sorted(candidates, key=lambda c: _score(c, geo, loss_fn))
    best_idx, best_loss = ranked[0], np.inf
    for start in ranked[:N_STARTS]:
        result = _swap_search(start, geo, loss_fn)
        loss = _score(result, geo, loss_fn)
        if loss < best_loss:
            best_idx, best_loss = result, loss
    return best_idx


def _corner_loss(
    r_cand: np.ndarray, omega_cand: np.ndarray, target_r: float, target_omega: float
) -> np.ndarray:
    """Corner loss (report App. A): weighted distance to a (r, omega) target."""
    return (
        np.abs(r_cand - target_r) / R_TOL
        + np.abs(omega_cand - target_omega) / OMEGA_TOL
    )


def _corner_loss_fn(target_r: float, target_omega: float) -> LossFn:
    """Bind one corner's (r, omega) target to :func:`_corner_loss`."""
    return partial(_corner_loss, target_r=target_r, target_omega=target_omega)


def _ten_match_loss(
    r_cand: np.ndarray, omega_cand: np.ndarray, rho: float, r_gl: float, neff_gl: float
) -> np.ndarray:
    """Ten-patient match loss: reproduce good-low coverage and effective support."""
    denom = 1.0 + 15.0 * rho + 90.0 * rho * omega_cand  # g=10, m=16
    neff_cand = np.where(denom > 0, 160.0 / np.maximum(denom, 1e-12), np.nan)
    r_term = np.abs(r_cand - r_gl) / R_TOL
    neff_term = np.where(
        denom > 0, np.abs(neff_cand / neff_gl - 1.0) / NEFF_REL_TOL, np.inf
    )
    return r_term + neff_term


def _ten_match_loss_fn(rho: float, r_gl: float, neff_gl: float) -> LossFn:
    """Bind the good-low target to :func:`_ten_match_loss`."""
    return partial(_ten_match_loss, rho=rho, r_gl=r_gl, neff_gl=neff_gl)


def _corner_targets(
    candidates: list[np.ndarray], geo: ClassGeometry
) -> dict[str, tuple[float, float]]:
    """Median-centred (r, omega) targets for the four coverage x similarity corners."""
    rs = np.array([r_of(geo.d_pool, c) for c in candidates])
    omegas = np.array([omega_of(geo, c) for c in candidates])
    r_med, omega_med = float(np.median(rs)), float(np.median(omegas))
    r_good, r_poor = r_med - TARGET_R_GAP / 2.0, r_med + TARGET_R_GAP / 2.0
    omega_low, omega_high = (
        omega_med - TARGET_OMEGA_GAP / 2.0,
        omega_med + TARGET_OMEGA_GAP / 2.0,
    )
    return {
        "good_low": (r_good, omega_low),
        "poor_low": (r_poor, omega_low),
        "good_high": (r_good, omega_high),
        "poor_high": (r_poor, omega_high),
    }


def _cohort_result(idx: np.ndarray, geo: ClassGeometry, g: int, m: int) -> CohortResult:
    omega = omega_of(geo, idx)
    return CohortResult(
        patients=[geo.pool[i] for i in idx],
        r_train=r_of(geo.d_pool, idx),
        r_val=r_of(geo.d_val, idx),
        omega=omega,
        neff=neff_of(g, m, geo.rho, omega),
    )


def search_class_draw(
    split_idx: int, draw_idx: int, c_idx: int, geo: ClassGeometry
) -> dict[str, CohortResult]:
    """All five cohorts of one (split, class, draw); ``c_idx`` is canonical-sorted."""
    seed5 = derive_draw_seed(SEARCH_BASE_SEED, split_idx, 5, 32, draw_idx, c_idx)
    candidates5 = _build_candidates(geo, 5, seed5)
    corners = _corner_targets(candidates5, geo)

    results: dict[str, CohortResult] = {}
    for name, target in corners.items():
        idx = _search(candidates5, geo, _corner_loss_fn(*target))
        results[name] = _cohort_result(idx, geo, 5, 32)

    good_low = results["good_low"]
    neff_gl = neff_of(5, 32, geo.rho, good_low.omega)
    seed10 = derive_draw_seed(SEARCH_BASE_SEED, split_idx, 10, 16, draw_idx, c_idx)
    candidates10 = _build_candidates(geo, 10, seed10)
    loss10 = (
        _ten_match_loss_fn(geo.rho, good_low.r_train, neff_gl)
        if neff_gl is not None
        else (lambda r_cand, omega_cand: np.full(len(r_cand), np.inf))
    )
    idx10 = _search(candidates10, geo, loss10)
    results["ten_match"] = _cohort_result(idx10, geo, 10, 16)
    return results


def search_split_draw(
    split_idx: int,
    draw_idx: int,
    canonical_names: list[str],
    geometry_by_class: dict[str, ClassGeometry],
) -> dict[str, dict[str, CohortResult]]:
    """Every canonical class's five cohorts for one (split, draw)."""
    return {
        c_name: search_class_draw(split_idx, draw_idx, c_idx, geometry_by_class[c_name])
        for c_idx, c_name in enumerate(canonical_names)
    }
