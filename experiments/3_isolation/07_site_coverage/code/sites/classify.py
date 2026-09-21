"""Interpretation classifier for the two prespecified site-coverage contrasts."""

from __future__ import annotations

from sites import THRESHOLD_PP

__all__ = ["classify"]


def _above_threshold(ci: tuple[float, float]) -> bool:
    """Whether an interval sits entirely above the practical threshold."""
    return ci[0] > THRESHOLD_PP


def _within_threshold(ci: tuple[float, float]) -> bool:
    """Whether an interval sits entirely within +/- the practical threshold."""
    return ci[0] >= -THRESHOLD_PP and ci[1] <= THRESHOLD_PP


def classify(ds_ci: tuple[float, float], bw_ci: tuple[float, float]) -> str:
    """Interpretation label (report Table "accounts")."""
    if _above_threshold(ds_ci) and _within_threshold(bw_ci):
        return "site_coverage"
    if _within_threshold(ds_ci) and _above_threshold(bw_ci):
        return "within_site_patient_variation"
    if _above_threshold(ds_ci) and _above_threshold(bw_ci):
        return "both"
    return "inconclusive"
