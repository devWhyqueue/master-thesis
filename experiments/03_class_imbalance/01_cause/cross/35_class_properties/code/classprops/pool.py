"""Per-class recall stacks, allocation covariates, and observation assembly (0 new fits).

One "observation" is a (dataset, pool, rho, arm, fit, class) tuple: the class's own recall change
Delta_fc = recall_c(arm, f) - recall_c(r1, same split/draw), its allocation z_fc, and the (split,
class) index needed to attach the cross-fitted headroom/margin covariates in ``classprops.model``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.common import read_run_record

from assignment._io import paths_by_split, require_record

from permutation.model import z_of_counts

from prevalence import patients_per_class

from sites import allocation_dir
from sites.recall import contexts

from neighbours.accuracy import recall_stack

from breadth.analyze.canonical import canonical_permutation

from centre import N_DRAWS, N_SPLITS

from spectrum import baseline_config

from classprops import ArmSpec

__all__ = [
    "Observations",
    "class_recall_stack",
    "build_pool",
    "observations_from_arrays",
    "pooled_ba",
    "arm_ba_delta",
    "read_r1_stack",
    "read_test_margins",
]

FIT_SPLIT: np.ndarray = np.repeat(np.arange(N_SPLITS), N_DRAWS)


def class_recall_stack(
    source_config: dict[str, Any], names: list[str], arm: str
) -> np.ndarray:
    """(F, C, R) per-class patient-macro recall in percent for every stored fit of one arm."""
    ctxs = contexts(source_config)
    paths = paths_by_split(source_config)
    perms = {s: canonical_permutation(source_config, s, names) for s in range(N_SPLITS)}
    keys = [(s, d) for s in range(N_SPLITS) for d in range(N_DRAWS)]
    return (
        recall_stack(
            [allocation_dir(paths[s], arm, d) for s, d in keys],
            [ctxs[s] for s, _ in keys],
            [perms[s] for s, _ in keys],
            len(names),
        )
        * 100.0
    )


def _counts_for_arm(
    source_config: dict[str, Any], names: list[str], arm: str, field: str
) -> np.ndarray:
    """(F, C) one stored count field (``class_counts`` or ``prior_counts``) for every fit."""
    paths = paths_by_split(source_config)
    rows = [
        [require_record(allocation_dir(paths[s], arm, d))[field][c] for c in names]
        for s in range(N_SPLITS)
        for d in range(N_DRAWS)
    ]
    return np.asarray(rows, dtype=np.float64)


def z_for_arm(
    source_config: dict[str, Any], names: list[str], arm: str, field: str
) -> np.ndarray:
    """(F, C) log2 allocation deviation from balanced, from one stored count field."""
    g = patients_per_class(source_config)
    return z_of_counts(_counts_for_arm(source_config, names, arm, field), g)


class Observations(NamedTuple):
    """One (dataset, pool, rho) pool's stacked (arm, fit, class) observations."""

    delta: np.ndarray  # (N, R) own-class recall change vs r1
    z: np.ndarray  # (N,) allocation, point value (no bootstrap variation)
    split_idx: np.ndarray  # (N,) int, the fit's split
    class_idx: np.ndarray  # (N,) int, index into names
    draw_weight: np.ndarray  # (N, R) draw-resampling weight, already divided by K


def build_pool(
    config: dict[str, Any],
    names: list[str],
    dataset: str,
    plan: list[ArmSpec],
    r1_stack: np.ndarray,
    w_draw: np.ndarray,
) -> Observations:
    """Stack every arm in ``plan`` into one pool's flat (arm, fit, class) observations.

    ``r1_stack`` (F, C, R) and ``w_draw`` (F, R) are shared across every pool of one dataset (same
    r1 baseline and the same draw-resampling weights), computed once by the caller.
    """
    k = len(names)
    n_fits, n_classes, n_replicates = r1_stack.shape
    class_idx_block = np.tile(np.arange(k), n_fits)
    split_idx_block = np.repeat(FIT_SPLIT, k)
    deltas, zs = [], []
    for arm, source_key, field in plan:
        source_config = baseline_config(config, source_key)
        stack = class_recall_stack(source_config, names, arm)  # (F, C, R)
        z = z_for_arm(source_config, names, arm, field)  # (F, C)
        deltas.append((stack - r1_stack).reshape(n_fits * n_classes, n_replicates))
        zs.append(z.reshape(n_fits * n_classes))
    n_arms = len(plan)
    weight_block = np.repeat(w_draw, k, axis=0) / k  # (F*C, R)
    return Observations(
        delta=np.concatenate(deltas, axis=0),
        z=np.concatenate(zs, axis=0),
        split_idx=np.tile(split_idx_block, n_arms),
        class_idx=np.tile(class_idx_block, n_arms),
        draw_weight=np.tile(weight_block, (n_arms, 1)),
    )


def observations_from_arrays(
    arrays: dict[str, np.ndarray], prefix: str
) -> Observations:
    """Reconstruct one pool's :class:`Observations` from its namespaced extract arrays."""
    return Observations(
        delta=arrays[f"{prefix}_delta"],
        z=arrays[f"{prefix}_z"],
        split_idx=arrays[f"{prefix}_split_idx"].astype(np.int64),
        class_idx=arrays[f"{prefix}_class_idx"].astype(np.int64),
        draw_weight=arrays[f"{prefix}_weight"],
    )


def pooled_ba(stack: np.ndarray, w_draw: np.ndarray) -> np.ndarray:
    """(R,): draw-weighted pooled macro balanced accuracy of a (F, C, R) recall stack."""
    macro = stack.mean(axis=1)  # (F, R)
    return (w_draw * macro).sum(axis=0) / w_draw.sum(axis=0)


def arm_ba_delta(
    config: dict[str, Any],
    names: list[str],
    arm: str,
    source_key: str,
    r1_stack: np.ndarray,
    w_draw: np.ndarray,
) -> np.ndarray:
    """(R,) pooled BA damage of one arm vs r1: -mean_c pooled(delta_c). Sanity-check helper."""
    source_config = baseline_config(config, source_key)
    stack = class_recall_stack(source_config, names, arm)
    return pooled_ba(r1_stack, w_draw) - pooled_ba(stack, w_draw)


def read_r1_stack(config: dict[str, Any], names: list[str]) -> np.ndarray:
    """(F, C, R) r1's own per-class recall, the shared baseline for every pool."""
    return class_recall_stack(
        baseline_config(config, "prevalence_outputs"), names, "r1"
    )


def _draw_margins(
    paths: dict[int, dict[str, Path]],
    arm: str,
    split: int,
    draw: int,
    inv_perm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One draw's per-test-patch margin and canonical class index."""
    rec = read_run_record(
        allocation_dir(paths[split], arm, draw),
        splits=("test",),
        array_fields=("labels", "probabilities"),
    )
    if rec is None:
        raise RuntimeError(f"Missing run record at split {split}, draw {draw}")
    test = rec["splits"]["test"]
    labels = np.asarray(test["labels"])
    floor = np.finfo(np.float64).tiny
    log_probs = np.log(np.maximum(np.asarray(test["probabilities"]), floor))
    rows = np.arange(len(labels))
    true_logp = log_probs[rows, labels]
    masked = log_probs.copy()
    masked[rows, labels] = -np.inf
    return true_logp - masked.max(axis=1), inv_perm[labels]


def read_test_margins(
    source_config: dict[str, Any],
    names: list[str],
    arm: str,
    split: int,
    ctx: BootstrapContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every test-patch margin, canonical class index, and row_patient index for one split.

    Pools every draw's copy of that split's fixed test set (only the fitted probabilities differ
    per draw); row order matches ``ctx``'s own identity frame (``sites.recall.contexts``), so its
    ``weights.row_patient`` (one draw's length) tiles across draws unchanged.
    """
    paths = paths_by_split(source_config)
    inv_perm = np.argsort(canonical_permutation(source_config, split, names))
    pairs = [_draw_margins(paths, arm, split, d, inv_perm) for d in range(N_DRAWS)]
    row_patient = np.tile(ctx.weights.row_patient, N_DRAWS)
    return (
        np.concatenate([p[0] for p in pairs]),
        np.concatenate([p[1] for p in pairs]),
        row_patient,
    )
