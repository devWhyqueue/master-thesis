"""Shared helpers of the analyze stage: arms, method families, CE anchors, selected recoveries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names, canonical_permutation

from sites.recall import contexts

from centre import N_DRAWS, N_SPLITS
from centre.analyze import arm_accuracy, pooled

from decomposition.model import draw_weights

from mitigation import STAGE1_METHODS, STAGE2_METHODS

from analyze.bootstrap import selected_ba
from analyze.paths import Shard, result_paths, run_dir
from analyze.select import select, selected_frequency

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
