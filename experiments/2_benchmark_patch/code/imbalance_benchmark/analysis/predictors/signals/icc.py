"""Descriptive ICC and effective-support diagnostics (report Eqs. icc, neff)."""

from __future__ import annotations

import numpy as np

__all__ = ["icc_estimate", "descriptive_support"]


def icc_estimate(scores: np.ndarray, case_ids: np.ndarray) -> float | None:
    """Eq. (icc): one-way ANOVA ICC with the unequal-cluster-size correction."""
    if len(scores) == 0:
        return None
    cases, inverse, counts = np.unique(
        case_ids, return_inverse=True, return_counts=True
    )
    g, n = len(cases), len(scores)
    if g < 2 or n - g <= 0:
        return None
    grand_mean = scores.mean()
    case_means = np.zeros(g)
    np.add.at(case_means, inverse, scores)
    case_means /= counts
    ss_between = float(np.sum(counts * (case_means - grand_mean) ** 2))
    ss_within = float(np.sum((scores - case_means[inverse]) ** 2))
    ms_between, ms_within = ss_between / (g - 1), ss_within / (n - g)
    m_tilde = (n - np.sum(counts**2) / n) / (g - 1)
    denom = ms_between + (m_tilde - 1) * ms_within
    if denom == 0:
        return 0.0
    return float(np.clip((ms_between - ms_within) / denom, 0.0, 1.0))


def descriptive_support(
    counts: np.ndarray, icc: float | None
) -> dict[str, float | None]:
    """Eq. (neff): DE_c and N_eff,c from the exact case profile and the ICC estimate."""
    n = int(counts.sum())
    m_star = float(np.sum(counts**2) / n) if n else 0.0
    if icc is None or n == 0:
        return {"icc": icc, "m_star": m_star, "n_eff": None}
    design_effect = 1.0 + (m_star - 1.0) * icc
    return {
        "icc": icc,
        "m_star": m_star,
        "n_eff": float(n / design_effect) if design_effect else None,
    }
