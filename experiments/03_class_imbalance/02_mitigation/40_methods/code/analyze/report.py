"""Pack pooled distributions into ``analysis.json``'s estimates and write it out."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from breadth.analyze.secondary import pack_estimate

from centre import N_DRAWS, N_SPLITS

__all__ = [
    "Distributions",
    "method_summary",
    "pack_methods",
    "family_share",
    "write_analysis",
]

RECOVERS_PP = 1.0


class Distributions(NamedTuple):
    """Every pooled (R,) distribution ``pack_methods`` reports, keyed by CE arm or method name."""

    ba: dict[str, np.ndarray]
    damage: np.ndarray
    ba_selected: dict[str, np.ndarray]
    recovery: dict[str, np.ndarray]
    share: dict[str, np.ndarray]
    frequency: dict[str, dict[str, int]]


def method_summary(
    family: str,
    selected_est: dict[str, float],
    recovery_est: dict[str, float],
    frequency: dict[str, int],
    r1_point: float,
) -> dict[str, Any]:
    """One method's recovery verdicts (pre-stated 1 pp / CI > 0 and full-recovery readings)."""
    return {
        "family": family,
        "selected_param_frequency": frequency,
        "recovers": bool(
            recovery_est["point"] >= RECOVERS_PP and recovery_est["ci_2_5"] > 0.0
        ),
        "full_recovery": bool(
            selected_est["ci_2_5"] <= r1_point <= selected_est["ci_97_5"]
        ),
    }


def _pack_one(
    m: str, dists: Distributions, family_of: dict[str, str], ce_r1: str
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, Any]]:
    selected_est = pack_estimate(dists.ba_selected[m])
    recovery_est = pack_estimate(dists.recovery[m])
    share_est = pack_estimate(dists.share[m])
    summary = method_summary(
        family_of[m], selected_est, recovery_est, dists.frequency[m], dists.ba[ce_r1][0]
    )
    return selected_est, recovery_est, share_est, summary


def pack_methods(
    methods: tuple[str, ...],
    family_of: dict[str, str],
    dists: Distributions,
    ce_r1: str,
    ce_r100: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every estimate (CE anchors, damage, and each method's selected/recovery/share) plus per-method verdicts."""
    estimates = {
        "arm_r1_ce": pack_estimate(dists.ba[ce_r1]),
        "arm_r100_ce": pack_estimate(dists.ba[ce_r100]),
        "damage": pack_estimate(dists.damage),
    }
    methods_out: dict[str, Any] = {}
    for m in methods:
        selected_est, recovery_est, share_est, summary = _pack_one(
            m, dists, family_of, ce_r1
        )
        estimates[f"selected_{m}"] = selected_est
        estimates[f"recovery_{m}"] = recovery_est
        estimates[f"share_{m}"] = share_est
        methods_out[m] = summary
    return estimates, methods_out


def family_share(
    methods: tuple[str, ...],
    family_of: dict[str, str],
    share_dist: dict[str, np.ndarray],
) -> dict[str, float | None]:
    """Mean recovered share of damage, by stage-one vs stage-two method family."""
    families: dict[str, list[float]] = {"stage1": [], "stage2": []}
    for m in methods:
        families[family_of[m]].append(float(share_dist[m][0]))
    return {fam: float(np.mean(v)) if v else None for fam, v in families.items()}


def write_analysis(
    config: dict[str, Any],
    estimates: dict[str, Any],
    methods_out: dict[str, Any],
    shares: dict[str, float | None],
    thirds: dict[str, dict[str, float]],
    quality: dict[str, dict[str, dict[str, float]]],
    exp26_damage: dict[str, float],
) -> Path:
    """Write the exp-41 analysis.json."""
    path = output_root(config) / "data" / "analysis.json"
    write_json(
        path,
        {
            "shards": {"n_splits": N_SPLITS, "n_draws": N_DRAWS},
            "estimates": estimates,
            "methods": methods_out,
            "family_share": shares,
            "rank_recall": thirds,
            "probability_quality": quality,
            "exp26_damage_reference": exp26_damage,
        },
    )
    return path
