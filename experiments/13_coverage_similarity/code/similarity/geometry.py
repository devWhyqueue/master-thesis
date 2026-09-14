"""Per (split, class) pool geometry: precomputed distances and Gram deviations."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from neighbours.coverage import _patients_of
from neighbours.embedding import cosine_distances

from coverage_redundancy.cohorts import _build_split_context

__all__ = [
    "SplitInputs",
    "ClassGeometry",
    "build_split_geometry",
    "r_of",
    "omega_of",
    "neff_of",
]


class SplitInputs(NamedTuple):
    """Cross-split state every split's geometry and census stage shares."""

    means: dict[tuple[str, str], np.ndarray]
    class_names: list[str]
    rho_by_class: dict[str, float]
    raw_full: np.ndarray


class ClassGeometry(NamedTuple):
    """One (split, class)'s pool geometry for the cohort search (report App. A)."""

    pool: list[str]
    d_pool: np.ndarray  # (P, P) cosine distance among eligible pool patients
    d_val: np.ndarray  # (V, P) cosine distance from validation patients to the pool
    gram: np.ndarray  # (P, P) pool-mean-centred deviation Gram matrix
    tau2: float
    rho: float


def _stack(
    embeddings: dict[tuple[str, str], np.ndarray], patients: list[str], c_name: str
) -> np.ndarray:
    return np.stack([embeddings[(p, c_name)] for p in patients])


def build_split_geometry(
    config: dict[str, Any],
    split_idx: int,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    shared: SplitInputs,
) -> dict[str, ClassGeometry]:
    """Every class's pool geometry for one split, guarded against exp-6's ICC."""
    ctx = _build_split_context(config, split_idx, full_df, train_df, *shared)
    out: dict[str, ClassGeometry] = {}
    for c_name in ctx.class_names:
        pool = ctx.eligible_by_class[c_name]
        val = _patients_of(full_df, "validation", c_name)
        e_pool = _stack(ctx.embeddings, pool, c_name)
        e_val = _stack(ctx.embeddings, val, c_name)
        # omega's tau2 is in raw feature units, so deviations must use raw means.
        raw_pool = _stack(ctx.means, pool, c_name)
        dev = raw_pool - raw_pool.mean(axis=0)
        out[c_name] = ClassGeometry(
            pool=pool,
            d_pool=cosine_distances(e_pool, e_pool),
            d_val=cosine_distances(e_val, e_pool),
            gram=dev @ dev.T,
            tau2=ctx.tau2_by_class[c_name],
            rho=ctx.rho_by_class[c_name],
        )
    return out


def r_of(d_ref_pool: np.ndarray, idx: np.ndarray) -> float:
    """Coverage distance r (Eq. coverage): mean nearest-cohort distance over the reference axis."""
    return float(np.mean(np.min(d_ref_pool[:, idx], axis=1)))


def omega_of(geo: ClassGeometry, idx: np.ndarray) -> float:
    """Between-patient similarity omega (Eq. omega), from the precomputed pool Gram matrix."""
    if geo.tau2 <= 0:
        return 0.0
    g = len(idx)
    sub = geo.gram[np.ix_(idx, idx)]
    off_diag_sum = float(sub.sum() - np.trace(sub))
    return off_diag_sum / (g * (g - 1) * geo.tau2)


def neff_of(g: int, m: int, rho: float, omega: float) -> float | None:
    """Similarity-adjusted effective support Neff^omega (Eq. support); None if infeasible."""
    denom = 1.0 + (m - 1.0) * rho + m * (g - 1.0) * rho * omega
    if denom <= 0:
        return None
    return g * m / denom
