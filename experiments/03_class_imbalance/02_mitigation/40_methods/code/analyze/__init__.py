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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names, canonical_permutation
from breadth.analyze.secondary import pack_estimate

from sites.recall import contexts

from centre import N_DRAWS, N_SPLITS
from centre.analyze import arm_accuracy, pooled

from decomposition.model import draw_weights

from prevalence.analyze import _thirds

from mitigation import STAGE1_METHODS, STAGE2_METHODS

from analyze.bootstrap import quality_distributions, selected_ba, selected_thirds
from analyze.figure import recovery_figure
from analyze.paths import Shard, fixed_dirs, result_paths, run_dir
from analyze.report import Distributions, family_share, pack_methods, write_analysis
from analyze.select import select, selected_frequency

__all__ = ["run_analyze"]

ARMS: tuple[str, ...] = ("r50", "r100", "N")
CE_R1 = "r1_ce"
METHODS: tuple[str, ...] = (
    tuple(m for m in STAGE1_METHODS if m != "ce") + STAGE2_METHODS
)
FAMILY: dict[str, str] = {
    **{m: "stage1" for m in STAGE1_METHODS if m != "ce"},
    **{m: "stage2" for m in STAGE2_METHODS},
}
# exp-26 (report/26_damage_bracs.tex): BRACS r1->r100 damage under a frozen logreg
# readout, not LoRA -- an external reference, no fits spent reproducing it here.
EXP26_DAMAGE = {"point": 7.65, "ci_2_5": 5.77, "ci_97_5": 9.78}


def _ce_arm(arm: str) -> str:
    """CE damage anchor's arm label for one imbalance arm, e.g. ``r100`` -> ``r100_ce``."""
    return f"{arm}_ce"


def _ce_ba_and_weights(
    config: dict[str, Any], names: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Pool ``r1_ce`` and every arm's own CE damage anchor under one shared bootstrap weight."""
    ce_acc = arm_accuracy(
        config, names, arms=(CE_R1,) + tuple(_ce_arm(a) for a in ARMS)
    )
    n_replicates = next(iter(ce_acc.values())).shape[-1]
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split, N_DRAWS, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {arm: pooled(a, w) for arm, a in ce_acc.items()}, w


def _recoveries(
    config: dict[str, Any], paths: dict[int, dict[str, Path]], arm: str
) -> dict[str, dict[str, Any]]:
    out = {}
    for m in METHODS:
        selection = select(config, paths, arm, FAMILY[m], m)
        out[m] = {"selection": selection, "frequency": selected_frequency(selection)}
    return out


def _selected_dirs(
    paths: dict[int, dict[str, Path]],
    recoveries: dict[str, dict[str, Any]],
    arm: str,
) -> dict[str, dict[Shard, Path]]:
    return {
        m: {
            key: run_dir(paths[key[0]], arm, m, param, key[1])
            for key, param in r["selection"].items()
        }
        for m, r in recoveries.items()
    }


def _recovery_distributions(
    ctxs: dict[int, Any],
    perms: dict[int, np.ndarray],
    n_classes: int,
    selected_dirs: dict[str, dict[Shard, Path]],
    ba: dict[str, np.ndarray],
    damage: np.ndarray,
    w: np.ndarray,
    ce_arm: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    ba_selected = {
        m: pooled(selected_ba(d, ctxs, perms, n_classes), w)
        for m, d in selected_dirs.items()
    }
    recovery_dist = {m: ba_selected[m] - ba[ce_arm] for m in METHODS}
    share_dist = {m: recovery_dist[m] / damage for m in METHODS}
    return ba_selected, recovery_dist, share_dist


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


def _setup(
    config: dict[str, Any],
) -> tuple[
    list[str], int, dict[int, dict[str, Path]], dict[int, Any], dict[int, np.ndarray]
]:
    names = canonical_class_names(config)
    paths = result_paths(config)
    ctxs = contexts(config)
    perms = {s: canonical_permutation(config, s, names) for s in range(N_SPLITS)}
    return names, len(names), paths, ctxs, perms


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
    arms_out = {
        arm: _analyze_arm(config, arm, names, n_classes, paths, ctxs, perms, ba, w)
        for arm in ARMS
    }
    return write_analysis(config, arms_out, EXP26_DAMAGE)
