"""Paired encoder contrast (exp-42): ``delta X = X(this config) - X(reference config)``.

Both encoders share the same bootstrap weights, so each delta is a paired difference of
pooled (R,) distributions. A quantity is encoder-dependent only if its 95% interval excludes
0 and ``|delta| >= ENCODER_PP``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.common import find_repo_root, load_config
from scipy.stats import rankdata

from breadth.analyze.secondary import pack_estimate

from analyze.balanced import balanced_gains
from analyze.common import (
    ARMS,
    CE_R1,
    _ce_arm,
    _ce_ba_and_weights,
    _recoveries,
    _recovery_distributions,
    _selected_dirs,
    _setup,
)
from analyze.report import RECOVERS_PP

__all__ = [
    "EncoderDistributions",
    "encoder_contrast",
    "check_same_cohort",
    "contrast_block",
]

ENCODER_PP = 1.0


class EncoderDistributions(NamedTuple):
    """One encoder's pooled (R,) distributions: CE r1 BA, per-arm damage/recovery, per-method gain."""

    ba_r1: np.ndarray
    damage: dict[str, np.ndarray]
    recovery: dict[str, dict[str, np.ndarray]]
    gain: dict[str, np.ndarray]


def check_same_cohort(
    names: tuple[list[str], list[str]],
    case_ids: tuple[dict[int, np.ndarray], dict[int, np.ndarray]],
) -> None:
    """Raise unless both configs share class names and per-split test identities."""
    if names[0] != names[1]:
        raise ValueError("encoder contrast: class names differ between configs")
    a, b = case_ids
    if a.keys() != b.keys() or any(not np.array_equal(a[s], b[s]) for s in a):
        raise ValueError("encoder contrast: test identities differ between configs")


def _delta(this: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    est = pack_estimate(this - ref)
    est["encoder_dependent"] = bool(
        (est["ci_2_5"] > 0.0 or est["ci_97_5"] < 0.0)
        and abs(est["point"]) >= ENCODER_PP
    )
    return est


def _recovers(dist: np.ndarray) -> bool:
    est = pack_estimate(dist)
    return bool(est["point"] >= RECOVERS_PP and est["ci_2_5"] > 0.0)


def _arm_contrast(
    this: EncoderDistributions, ref: EncoderDistributions, arm: str
) -> dict[str, Any]:
    methods = list(this.recovery[arm])
    deltas: dict[str, Any] = {
        "ba_r1_ce": _delta(this.ba_r1, ref.ba_r1),
        "damage": _delta(this.damage[arm], ref.damage[arm]),
    }
    for m in methods:
        deltas[f"recovery_{m}"] = _delta(this.recovery[arm][m], ref.recovery[arm][m])
    for m, g in this.gain.items():
        deltas[f"gain_{m}"] = _delta(g, ref.gain[m])
        specific = this.recovery[arm][m] - g
        deltas[f"specific_recovery_{m}"] = _delta(
            specific, ref.recovery[arm][m] - ref.gain[m]
        )
    points = [[float(e.recovery[arm][m][0]) for m in methods] for e in (this, ref)]
    rho = float(np.corrcoef(rankdata(points[0]), rankdata(points[1]))[0, 1])
    return {
        "delta": deltas,
        "verdict_agreement": {
            m: _recovers(this.recovery[arm][m]) == _recovers(ref.recovery[arm][m])
            for m in methods
        },
        "spearman_recovery": None if np.isnan(rho) else rho,
    }


def encoder_contrast(
    this: EncoderDistributions, ref: EncoderDistributions
) -> dict[str, dict[str, Any]]:
    """Per-arm paired deltas, encoder-dependence flags, verdict agreement, and rank correlation."""
    return {arm: _arm_contrast(this, ref, arm) for arm in this.recovery}


def _encoder_distributions(
    config: dict[str, Any],
) -> tuple[EncoderDistributions, list[str], dict[int, np.ndarray], np.ndarray]:
    """Pooled distributions one encoder's contrast needs, same recipe as ``run_analyze``."""
    names, n_classes, paths, ctxs, perms = _setup(config)
    ba, w = _ce_ba_and_weights(config, names)
    gains = balanced_gains(config, paths, ctxs, perms, ba[CE_R1], w)
    damage, recovery = {}, {}
    for arm in ARMS:
        damage[arm] = ba[CE_R1] - ba[_ce_arm(arm)]
        selected = _selected_dirs(paths, _recoveries(config, paths, arm), arm)
        recovery[arm] = _recovery_distributions(
            ctxs, perms, n_classes, selected, ba, damage[arm], w, _ce_arm(arm)
        )[1]
    dists = EncoderDistributions(
        ba[CE_R1], damage, recovery, {m: g["gain"] for m, g in gains.items()}
    )
    return dists, names, {s: c.case_ids for s, c in ctxs.items()}, w


def contrast_block(config: dict[str, Any], reference_path: str) -> dict[str, Any]:
    """Paired encoder contrast of ``config`` against the config at ``reference_path``."""
    reference = load_config(find_repo_root() / reference_path)
    this, names, ids, w = _encoder_distributions(config)
    ref, ref_names, ref_ids, ref_w = _encoder_distributions(reference)
    check_same_cohort((names, ref_names), (ids, ref_ids))
    if not np.array_equal(w, ref_w):
        raise ValueError("encoder contrast: bootstrap weights differ between configs")
    return encoder_contrast(this, ref)
