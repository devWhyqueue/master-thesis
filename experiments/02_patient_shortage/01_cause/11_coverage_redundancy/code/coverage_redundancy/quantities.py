"""Per-cohort quantities: tau^2's ICC identity, between-patient omega, Neff^omega."""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple

import numpy as np
import pandas as pd

from redundancy.correlate import _sample_split
from redundancy.estimator import (
    _between_sum_of_squares,
    _weighted_sums,
    cell_effective_support,
    weighted_icc,
)

from neighbours.coverage import _coverage_distance, _embedding_matrix, _patients_of

__all__ = ["SplitContext", "omega_similarity", "neff_omega", "class_tau2"]


class SplitContext(NamedTuple):
    """Everything one split's random and composition cohort assembly needs."""

    split_idx: int
    full_df: pd.DataFrame
    train_df: pd.DataFrame
    split_class_names: list[str]
    class_names: list[str]
    means: dict[tuple[str, str], np.ndarray]
    embeddings: dict[tuple[str, str], np.ndarray]
    eligible_by_class: dict[str, list[str]]
    tau2_by_class: dict[str, float]
    rho_by_class: dict[str, float]


def omega_similarity(
    means: dict[tuple[str, str], np.ndarray],
    patients: list[str],
    eligible: list[str],
    class_name: str,
    tau2: float,
) -> float:
    """Between-patient similarity omega_rc(S) (Eq. omega); 0 when tau^2 <= 0."""
    if tau2 <= 0:
        return 0.0
    mu = np.mean([means[(p, class_name)] for p in eligible], axis=0)
    deviations = np.stack([means[(p, class_name)] - mu for p in patients])
    g = len(patients)
    sum_of_squared_norms = float(np.sum(deviations**2))
    norm_of_summed_deviations = float(np.sum(deviations.sum(axis=0) ** 2))
    return (norm_of_summed_deviations - sum_of_squared_norms) / (g * (g - 1) * tau2)


def neff_omega(
    g: int,
    m: int,
    rho_bar: dict[str, float],
    omega: dict[str, float],
    class_names: Iterable[str],
) -> float:
    """Similarity-adjusted effective support Neff^omega (Eq. neff)."""
    values = []
    for c in class_names:
        rho = rho_bar[c]
        denom = 1.0 + (m - 1.0) * rho + m * (g - 1.0) * rho * omega[c]
        if denom <= 0:
            raise ValueError(f"Non-positive Neff^omega denominator for class {c}")
        values.append(g * m / denom)
    return float(np.mean(values))


def class_tau2(
    train_df: pd.DataFrame, class_names: list[str], split_idx: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Between-patient variance tau^2_rc and its ICC guard value, per class.

    Reproduces exp-6's own sample (:func:`redundancy.correlate._sample_split`)
    so tau^2 = (B - W) / m_tilde and the ones-weighted ICC (B - W) /
    (B + (m_tilde - 1) W) are computed on the identical patients exp-6 used.
    """
    split_stats = _sample_split(train_df, class_names, split_idx)
    tau2: dict[str, float] = {}
    icc_check: dict[str, float] = {}
    for class_name, measures in split_stats.items():
        stats = measures["full"]
        w = np.ones((1, len(stats.cases)))
        sums = _weighted_sums(stats, w)
        ssb = _between_sum_of_squares(stats, sums.wn, sums.total_n)
        between_ms = ssb / (sums.h - 1.0)
        within_ms = sums.ssw_weighted / (sums.total_n - sums.h)
        m_tilde = (sums.total_n - sums.sum_wn2 / sums.total_n) / (sums.h - 1.0)
        tau2[class_name] = float((between_ms - within_ms)[0] / m_tilde[0])
        icc_check[class_name] = float(weighted_icc(stats, w)[0])
    return tau2, icc_check


def class_quantities(
    ctx: SplitContext, c_name: str, patients: list[str]
) -> dict[str, float]:
    """Between-patient similarity omega and coverage distance r for one class."""
    omega = omega_similarity(
        ctx.means,
        patients,
        ctx.eligible_by_class[c_name],
        c_name,
        ctx.tau2_by_class[c_name],
    )
    e_val = _embedding_matrix(
        ctx.embeddings, _patients_of(ctx.full_df, "validation", c_name), c_name
    )
    e_s = _embedding_matrix(ctx.embeddings, patients, c_name)
    r = _coverage_distance(e_val, e_s)
    return {"omega": omega, "r": r}


def cohort_record(
    ctx: SplitContext,
    source: str,
    g: int,
    m: int,
    allocation: str | None,
    draw: int,
    per_class: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Assemble one cohort's log Neff, log Neff^omega, and mean coverage distance."""
    rho_arr = np.array([[ctx.rho_by_class[c] for c in ctx.class_names]])
    neff = float(cell_effective_support(rho_arr, [(g, m)])[0, 0])
    omega_by_class = {c: per_class[c]["omega"] for c in ctx.class_names}
    neff_om = neff_omega(g, m, ctx.rho_by_class, omega_by_class, ctx.class_names)
    return {
        "source": source,
        "split": ctx.split_idx,
        "g": g,
        "m": m,
        "allocation": allocation,
        "draw": draw,
        "log_neff": float(np.log(neff)),
        "log_neff_omega": float(np.log(neff_om)),
        "r": float(np.mean([per_class[c]["r"] for c in ctx.class_names])),
        "omega_mean": float(np.mean([per_class[c]["omega"] for c in ctx.class_names])),
        "classes": per_class,
    }
