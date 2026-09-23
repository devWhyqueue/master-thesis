"""H3's subset selection: 500 random 7-class subsets plus one greedily grown from the pooled
confusion matrix of r1's own stored predictions. Evaluating a selected subset (``subset_point``)
lives in ``margin.subset_eval``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.common import read_run_record

from breadth import BOOTSTRAP_SEED, exp2_split_paths
from breadth.analyze.canonical import canonical_permutation

from sites import allocation_dir

from centre import N_DRAWS, N_SPLITS

from margin import SUBSET_K

__all__ = ["point_contexts", "random_subsets", "greedy_confused_subset"]


def point_contexts(config: dict[str, Any]) -> dict[int, BootstrapContext]:
    """One observed-only (no bootstrap) context per split, for H3's 500+ point-estimate subsets."""
    return {
        s: BootstrapContext(
            exp2_split_paths(config, s),
            is_mil=False,
            n_replicates=1,
            seed=BOOTSTRAP_SEED,
        )
        for s in range(N_SPLITS)
    }


def random_subsets(
    rng: np.random.Generator, canonical_names: list[str], n: int
) -> list[list[str]]:
    """``n`` random ``SUBSET_K``-class subsets of the canonical class list."""
    idx = np.arange(len(canonical_names))
    return [
        [canonical_names[i] for i in rng.choice(idx, size=SUBSET_K, replace=False)]
        for _ in range(n)
    ]


def _pooled_confusion(
    config: dict[str, Any],
    ds_paths: dict[int, dict[str, Path]],
    canonical_names: list[str],
) -> np.ndarray:
    """Pooled canonical (K, K) confusion count matrix of r1's own stored predictions."""
    k = len(canonical_names)
    counts = np.zeros((k, k), dtype=np.int64)
    for s in range(N_SPLITS):
        perm = canonical_permutation(config, s, canonical_names)
        inv = np.empty(k, dtype=np.int64)
        inv[perm] = np.arange(k)
        for d in range(N_DRAWS):
            rec = read_run_record(
                allocation_dir(ds_paths[s], "r1", d),
                splits=("test",),
                array_fields=("labels", "preds"),
            )
            if rec is None:
                raise RuntimeError(f"Missing r1 run record at split {s}, draw {d}")
            test = rec["splits"]["test"]
            true_c = inv[np.asarray(test["labels"])]
            pred_c = inv[np.asarray(test["preds"])]
            np.add.at(counts, (true_c, pred_c), 1)
    return counts


def _grow_confused_group(sym: np.ndarray, n_classes: int) -> list[int]:
    """Seed with the most-confused pair, then repeatedly add the class most confused with the
    group, by ``sym[c, g] = rate[c, g] + rate[g, c]``, until it holds ``SUBSET_K`` classes."""
    i, j = np.unravel_index(np.argmax(sym), sym.shape)
    group = [int(i), int(j)]
    while len(group) < SUBSET_K:
        scores = [
            (max(sym[c, g] for g in group), c)
            for c in range(n_classes)
            if c not in group
        ]
        group.append(max(scores)[1])
    return group


def greedy_confused_subset(
    config: dict[str, Any],
    ds_paths: dict[int, dict[str, Path]],
    canonical_names: list[str],
) -> list[str]:
    """Greedily grow a ``SUBSET_K``-class group by symmetric pairwise confusion rate (off-diagonal
    share of each class's own test rows)."""
    counts = _pooled_confusion(config, ds_paths, canonical_names)
    rate = counts / counts.sum(axis=1, keepdims=True)
    sym = rate + rate.T
    np.fill_diagonal(sym, -1.0)
    group = _grow_confused_group(sym, len(canonical_names))
    return [canonical_names[c] for c in group]
