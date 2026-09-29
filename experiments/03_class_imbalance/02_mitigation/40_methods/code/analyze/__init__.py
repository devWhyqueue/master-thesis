"""Analyze stage (exp-41): validation-selected recovery of each mitigation method,
at every imbalance arm the config grids (``ARMS``: r50, r100, N).

Per (split, draw) shard, the method's configured param with the best *validation*
patient-macro balanced accuracy is selected (``analyze.select``, never the test
set); its test-set patient-macro BA is then pooled across shards with the same
paired Bayesian bootstrap machinery exp-16/17/25 already use (``draw_weights``,
``BootstrapContext`` via ``sites.recall.contexts``). Recovery at a given arm is
read against ``r1_ce`` (undamaged, shared by every arm) and that arm's own
``{arm}_ce`` (damage anchor); ``centre.analyze.arm_accuracy``/``pooled`` and
``prevalence.analyze._thirds`` apply as-is (one arm name spans every shard); the
per-shard *varying* winning directory selection produces does not fit those
wrappers, so ``analyze.bootstrap`` reapplies the same underlying
recall/probability-quality primitives directly. All arms and methods share one
set of bootstrap replicate weights, so every contrast stays paired.

A method also gridded on ``r1`` (CUDA) gets an imbalance-specific recovery (``analyze.balanced``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root

from breadth.analyze.secondary import pack_estimate

from centre.analyze import pooled

from prevalence.analyze import _thirds

from analyze.balanced import balanced_gains, pack_balanced, specific_estimates
from analyze.bootstrap import quality_distributions, selected_thirds
from analyze.common import (
    ARMS,
    CE_R1,
    FAMILY,
    METHODS,
    _ce_arm,
    _ce_ba_and_weights,
    _recoveries,
    _recovery_distributions,
    _selected_dirs,
    _setup,
)
from analyze.figure import recovery_figure
from analyze.paths import Shard, fixed_dirs
from analyze.contrast import contrast_block
from analyze.report import Distributions, family_share, pack_methods, write_analysis

__all__ = ["run_analyze"]

# exp-26 (report/26_damage_bracs.tex): BRACS r1->r100 damage under a frozen logreg
# readout, not LoRA -- an external reference, no fits spent reproducing it here.
EXP26_DAMAGE = {"point": 7.65, "ci_2_5": 5.77, "ci_97_5": 9.78}


def _rank_recall(
    config: dict[str, Any],
    names: list[str],
    ctxs: dict[int, Any],
    selected_dirs: dict[str, dict[Shard, Path]],
    n_classes: int,
    ce_arm: str,
) -> dict[str, dict[str, float]]:
    ce_thirds = _thirds(config, (CE_R1, ce_arm), names)
    thirds = {
        arm: {rank: ce_thirds[rank][arm] for rank in ("head", "body", "tail")}
        for arm in (CE_R1, ce_arm)
    }
    thirds.update(
        {m: selected_thirds(d, ctxs, n_classes) for m, d in selected_dirs.items()}
    )
    return thirds


def _quality_estimates(
    paths: dict[int, dict[str, Path]],
    ctxs: dict[int, Any],
    selected_dirs: dict[str, dict[Shard, Path]],
    n_classes: int,
    w: np.ndarray,
    arm: str,
    ce_arm: str,
) -> dict[str, dict[str, dict[str, float]]]:
    dirs = {
        CE_R1: fixed_dirs(paths, "r1", "ce", None),
        ce_arm: fixed_dirs(paths, arm, "ce", None),
    }
    dirs.update(selected_dirs)
    return {
        label: {
            key: pack_estimate(pooled(dist, w))
            for key, dist in quality_distributions(d, ctxs, n_classes).items()
        }
        for label, d in dirs.items()
    }


def _analyze_arm(
    config: dict[str, Any],
    arm: str,
    names: list[str],
    n_classes: int,
    paths: dict[int, dict[str, Path]],
    ctxs: dict[int, Any],
    perms: dict[int, np.ndarray],
    ba: dict[str, np.ndarray],
    w: np.ndarray,
    gains: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Every reported quantity for one imbalance arm's recovery grid."""
    ce_arm = _ce_arm(arm)
    damage = ba[CE_R1] - ba[ce_arm]
    recoveries = _recoveries(config, paths, arm)
    selected_dirs = _selected_dirs(paths, recoveries, arm)
    ba_selected, recovery_dist, share_dist = _recovery_distributions(
        ctxs, perms, n_classes, selected_dirs, ba, damage, w, ce_arm
    )
    thirds = _rank_recall(config, names, ctxs, selected_dirs, n_classes, ce_arm)
    quality = _quality_estimates(paths, ctxs, selected_dirs, n_classes, w, arm, ce_arm)
    frequency = {m: r["frequency"] for m, r in recoveries.items()}
    dists = Distributions(ba, damage, ba_selected, recovery_dist, share_dist, frequency)

    estimates, methods_out = pack_methods(METHODS, FAMILY, dists, CE_R1, ce_arm)
    estimates.update(specific_estimates(gains, recovery_dist, damage))
    shares = family_share(METHODS, FAMILY, share_dist)
    recovery_figure(
        estimates,
        METHODS,
        output_root(config) / "figures" / f"recovery_{arm}.pdf",
        ce_arm=ce_arm,
        arm_label=arm,
    )
    return {
        "estimates": estimates,
        "methods": methods_out,
        "family_share": shares,
        "rank_recall": thirds,
        "probability_quality": quality,
    }


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool the CE anchors and every method's validation-selected recovery, per arm; write the analysis."""
    names, n_classes, paths, ctxs, perms = _setup(config)
    ba, w = _ce_ba_and_weights(config, names)
    gains = balanced_gains(config, paths, ctxs, perms, ba[CE_R1], w)
    arms_out = {
        arm: _analyze_arm(
            config, arm, names, n_classes, paths, ctxs, perms, ba, w, gains
        )
        for arm in ARMS
    }
    settings = config.get("analyze", {})
    extra = {}
    if "reference_config" in settings:
        extra["encoder_contrast"] = contrast_block(config, settings["reference_config"])
    return write_analysis(
        config,
        arms_out,
        settings.get("external_damage", EXP26_DAMAGE),
        pack_balanced(gains),
        extra,
    )
