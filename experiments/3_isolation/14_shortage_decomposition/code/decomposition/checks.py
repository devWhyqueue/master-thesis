"""The manipulation check: spread, separation, overlap, and transfer (report Sec. "check")."""

from __future__ import annotations

from typing import Any

import numpy as np

from decomposition import (
    OVERLAP_MIN_SHARE,
    OVERLAP_OMEGA_TOL,
    OVERLAP_R_FACTOR,
    SEPARATION_MAX_ABS_CORR,
    SPREAD_OMEGA_MIN,
    SPREAD_R_FACTOR,
    TRANSFER_MIN_RATIO,
)

__all__ = ["manipulation_check"]


def _designed(rows: list[dict[str, Any]], g: int) -> list[dict[str, Any]]:
    return [r for r in rows if r["g"] == g and r["r_level"] is not None]


def _mean_at_level(
    rows: list[dict[str, Any]], key: str, level_key: str, level: int
) -> float:
    values = [r[key] for r in rows if r[level_key] == level]
    return float(np.mean(values)) if values else float("nan")


def spread(rows: list[dict[str, Any]], g: int) -> dict[str, Any]:
    """Spread check: r and omega gap between the poorest and best designed level."""
    d_rows = _designed(rows, g)
    spread_r = _mean_at_level(d_rows, "r_train", "r_level", 2) - _mean_at_level(
        d_rows, "r_train", "r_level", 0
    )
    spread_omega = _mean_at_level(d_rows, "omega", "omega_level", 2) - _mean_at_level(
        d_rows, "omega", "omega_level", 0
    )
    mean_delta_prime = float(np.mean([r["delta_prime"] for r in d_rows]))
    passed = (
        spread_r >= SPREAD_R_FACTOR * mean_delta_prime
        and spread_omega >= SPREAD_OMEGA_MIN
    )
    return {
        "spread_r": spread_r,
        "spread_omega": spread_omega,
        "mean_delta_prime": mean_delta_prime,
        "pass": bool(passed),
    }


def _separation(rows: list[dict[str, Any]], g: int) -> dict[str, Any]:
    """Separation check: correlation of class-centred coverage distance and similarity."""
    d_rows = _designed(rows, g)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for r in d_rows:
        by_class.setdefault(r["class"], []).append(r)
    r_centred, o_centred = [], []
    for c_rows in by_class.values():
        r_vals = np.array([r["r_train"] for r in c_rows])
        o_vals = np.array([r["omega"] for r in c_rows])
        r_centred.append(r_vals - r_vals.mean())
        o_centred.append(o_vals - o_vals.mean())
    corr = float(
        np.corrcoef(np.concatenate(r_centred), np.concatenate(o_centred))[0, 1]
    )
    return {"correlation": corr, "pass": bool(abs(corr) <= SEPARATION_MAX_ABS_CORR)}


def _transfer(rows: list[dict[str, Any]], g: int, spread_r: float) -> dict[str, Any]:
    """Transfer check: the validation-patient coverage spread relative to the pool spread."""
    d_rows = _designed(rows, g)
    spread_val = _mean_at_level(d_rows, "r_val", "r_level", 2) - _mean_at_level(
        d_rows, "r_val", "r_level", 0
    )
    ratio = spread_val / spread_r if spread_r != 0 else float("nan")
    return {
        "spread_r_val": spread_val,
        "ratio": ratio,
        "pass": bool(ratio >= TRANSFER_MIN_RATIO),
    }


def _overlap_share(rows: list[dict[str, Any]], g: int) -> float:
    """Share of designed cells on target: within Delta'/4 of r and 0.02 of omega."""
    d_rows = _designed(rows, g)
    if not d_rows:
        return float("nan")
    on_target = [
        abs(r["r_train"] - r["r_target"]) <= OVERLAP_R_FACTOR * r["delta_prime"]
        and abs(r["omega"] - r["omega_target"]) <= OVERLAP_OMEGA_TOL
        for r in d_rows
    ]
    return float(np.mean(on_target))


def manipulation_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The four-condition manipulation check, evaluated separately for G=5 and G=10."""
    by_g: dict[str, Any] = {}
    overlap_pass = True
    for g in (5, 10):
        g_spread = spread(rows, g)
        share = _overlap_share(rows, g)
        overlap_pass = overlap_pass and share >= OVERLAP_MIN_SHARE
        by_g[str(g)] = {
            "spread": g_spread,
            "separation": _separation(rows, g),
            "transfer": _transfer(rows, g, g_spread["spread_r"]),
            "overlap_share": share,
        }
    passed = overlap_pass and all(
        by_g[str(g)][check]["pass"]
        for g in (5, 10)
        for check in ("spread", "separation", "transfer")
    )
    return {"pass": bool(passed), "overlap_pass": bool(overlap_pass), "by_g": by_g}
