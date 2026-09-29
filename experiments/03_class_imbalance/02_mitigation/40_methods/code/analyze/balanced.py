"""Balanced gain of a method also gridded on ``r1`` (CUDA, GCL, stage-two methods), and its imbalance-specific recovery.

``G = BA(m*, r1) - BA(CE, r1)`` is the method's gain without imbalance; per arm, ``R - G`` (the
damage CE suffers minus the damage the method suffers) separates a general training benefit
from a correction of the imbalance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from breadth.analyze.secondary import pack_estimate

from centre.analyze import pooled

from analyze.bootstrap import selected_ba
from analyze.paths import run_dir
from analyze.select import select, selected_frequency

__all__ = ["balanced_gains", "specific_estimates", "pack_balanced"]


def balanced_gains(
    config: dict[str, Any],
    paths: dict[int, dict[str, Path]],
    ctxs: dict[int, Any],
    perms: dict[int, np.ndarray],
    ba_r1_ce: np.ndarray,
    w: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Per non-CE method gridded on ``r1`` (either stage), its validation-selected gain over ``r1_ce``."""
    n_classes = len(next(iter(perms.values())))
    out = {}
    for stage, methods in config["mitigation"]["grid"]["r1"].items():
        for m in methods:
            if m != "ce":
                out[m] = _gain(
                    config, paths, (ctxs, perms, n_classes), stage, m, ba_r1_ce, w
                )
    return out


def _gain(
    config: dict[str, Any],
    paths: dict[int, dict[str, Path]],
    bootstrap: tuple[dict[int, Any], dict[int, np.ndarray], int],
    stage: str,
    m: str,
    ba_r1_ce: np.ndarray,
    w: np.ndarray,
) -> dict[str, Any]:
    """One method's validation-selected ``r1`` BA minus ``r1_ce``, with its param frequency."""
    ctxs, perms, n_classes = bootstrap
    selection = select(config, paths, "r1", stage, m)
    dirs = {k: run_dir(paths[k[0]], "r1", m, p, k[1]) for k, p in selection.items()}
    ba = pooled(selected_ba(dirs, ctxs, perms, n_classes), w)
    return {"gain": ba - ba_r1_ce, "frequency": selected_frequency(selection)}


def specific_estimates(
    gains: dict[str, dict[str, Any]],
    recovery_dist: dict[str, np.ndarray],
    damage: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Imbalance-specific recovery ``R - G`` and its share of the CE damage, per gridded method."""
    out = {}
    for m, g in gains.items():
        specific = recovery_dist[m] - g["gain"]
        out[f"specific_recovery_{m}"] = pack_estimate(specific)
        out[f"specific_share_{m}"] = pack_estimate(specific / damage)
    return out


def pack_balanced(gains: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each method's balanced gain estimate and its selected-param frequency."""
    return {
        m: {
            "gain": pack_estimate(g["gain"]),
            "selected_param_frequency": g["frequency"],
        }
        for m, g in gains.items()
    }
