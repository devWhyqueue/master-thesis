"""The manipulation check: spread, separation, overlap, and transfer, per patient count."""

from __future__ import annotations

from typing import Any

import numpy as np

from hull import (
    OVERLAP_H_FACTOR,
    OVERLAP_MIN_SHARE,
    OVERLAP_OMEGA_TOL,
    OVERLAP_R_FACTOR,
    SEPARATION_MAX_ABS_CORR,
    SPREAD_H_FULL_FACTOR,
    SPREAD_H_SHARED_FACTOR,
    SPREAD_R_FACTOR,
    TRANSFER_MIN_RATIO,
)

__all__ = ["manipulation_check"]

Row = dict[str, Any]
_TOP_HULL_LEVEL = {5: 1.0, 10: 0.5}


def _designed(rows: list[Row], g: int) -> list[Row]:
    return [r for r in rows if r["g"] == g and r["mean_level"] is not None]


def _level_gap(rows: list[Row], key: str, level_key: str, high: float) -> float:
    """Mean of ``key`` at level ``high`` minus its mean at level 0."""
    hi = [r[key] for r in rows if r[level_key] == high]
    lo = [r[key] for r in rows if r[level_key] == 0.0]
    return float(np.mean(hi) - np.mean(lo)) if hi and lo else float("nan")


def _spread(rows: list[Row], g: int) -> dict[str, Any]:
    """Mean-coverage gap, shared hull gap (0 to 1/2), and for G=5 the full hull gap (0 to 1)."""
    mean_dr = float(np.mean([r["delta_r"] for r in rows]))
    mean_dh = float(np.mean([r["delta_h"] for r in rows]))
    out: dict[str, Any] = {
        "gap_r": _level_gap(rows, "r_train", "mean_level", 1.0),
        "gap_h_shared": _level_gap(rows, "h_train", "hull_level", 0.5),
        "mean_delta_r": mean_dr,
        "mean_delta_h": mean_dh,
    }
    passed = (
        out["gap_r"] >= SPREAD_R_FACTOR * mean_dr
        and out["gap_h_shared"] >= SPREAD_H_SHARED_FACTOR * mean_dh
    )
    if g == 5:
        out["gap_h_full"] = _level_gap(rows, "h_train", "hull_level", 1.0)
        passed = passed and out["gap_h_full"] >= SPREAD_H_FULL_FACTOR * mean_dh
    out["pass"] = bool(passed)
    return out


def _class_centred(rows: list[Row], key: str) -> np.ndarray:
    by_class: dict[str, list[float]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r[key])
    return np.concatenate([np.asarray(v) - np.mean(v) for v in by_class.values()])


def _separation(rows: list[Row]) -> dict[str, Any]:
    """Class-centred correlations of hull residual with coverage distance and with similarity."""
    h = _class_centred(rows, "h_train")
    corr_r = float(np.corrcoef(h, _class_centred(rows, "r_train"))[0, 1])
    corr_omega = float(np.corrcoef(h, _class_centred(rows, "omega"))[0, 1])
    passed = max(abs(corr_r), abs(corr_omega)) <= SEPARATION_MAX_ABS_CORR
    return {"corr_h_r": corr_r, "corr_h_omega": corr_omega, "pass": bool(passed)}


def _on_target(r: Row) -> bool:
    return (
        abs(r["r_train"] - r["r_target"]) <= OVERLAP_R_FACTOR * abs(r["delta_r"])
        and abs(r["h_train"] - r["h_target"]) <= OVERLAP_H_FACTOR * abs(r["delta_h"])
        and abs(r["omega"] - r["omega_target"]) <= OVERLAP_OMEGA_TOL
    )


def _overlap_share(rows: list[Row]) -> float:
    """Share of cohorts in the cells shared by both patient counts (hull level <= 1/2) on target."""
    shared = [r for r in rows if r["hull_level"] <= 0.5]
    return float(np.mean([_on_target(r) for r in shared])) if shared else float("nan")


def _transfer(rows: list[Row], g: int) -> dict[str, Any]:
    """Validation-patient gaps relative to pool gaps, for coverage and for hull residual."""
    top = _TOP_HULL_LEVEL[g]
    ratio_r = _level_gap(rows, "r_val", "mean_level", 1.0) / _level_gap(
        rows, "r_train", "mean_level", 1.0
    )
    ratio_h = _level_gap(rows, "h_val", "hull_level", top) / _level_gap(
        rows, "h_train", "hull_level", top
    )
    passed = min(ratio_r, ratio_h) >= TRANSFER_MIN_RATIO
    return {"ratio_r": ratio_r, "ratio_h": ratio_h, "pass": bool(passed)}


def manipulation_check(rows: list[Row]) -> dict[str, Any]:
    """The four-condition manipulation check, evaluated separately for G=5 and G=10."""
    by_g: dict[str, Any] = {}
    for g in (5, 10):
        designed = _designed(rows, g)
        share = _overlap_share(designed)
        by_g[str(g)] = {
            "spread": _spread(designed, g),
            "separation": _separation(designed),
            "transfer": _transfer(designed, g),
            "overlap_share": share,
            "overlap_pass": bool(share >= OVERLAP_MIN_SHARE),
        }
    passed = all(
        v["overlap_pass"]
        and all(v[k]["pass"] for k in ("spread", "separation", "transfer"))
        for v in by_g.values()
    )
    return {"pass": bool(passed), "by_g": by_g}
