"""H3's subset evaluation: the same injection formula (``margin.inject``) restricted to a 7-class
subset of r1's stored probabilities, renormalised over the subset. Both the rho=1 (uniform share)
and rho=100 (random-assignment) decisions reuse ``injected_preds`` so the restricted-softmax
decision rule is identical for the baseline and the injected point -- the only caveat is that a
restricted softmax is not a retrained 7-class model. Subset selection lives in ``margin.subsets``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.common import read_run_record

from breadth.analyze.canonical import canonical_permutation
from breadth.analyze.secondary import _patient_macro_recalls

from sites import allocation_dir

from centre import N_DRAWS, N_SPLITS

from margin import SUBSET_K
from margin.inject import _PROBABILITY_FLOOR, injected_preds, rho_shares

__all__ = ["renormalize_subset", "load_r1_cache", "subset_point"]

DrawCache = dict[int, tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]]


def renormalize_subset(probs: np.ndarray, local_cols: np.ndarray) -> np.ndarray:
    """Restrict ``probs`` to ``local_cols`` and renormalise each row back to a distribution."""
    restricted = probs[:, local_cols]
    return restricted / restricted.sum(axis=1, keepdims=True)


def load_r1_cache(
    config: dict[str, Any],
    ds_paths: dict[int, dict[str, Path]],
    canonical_names: list[str],
) -> DrawCache:
    """Per split: canonical permutation and every draw's (labels, probs), loaded once.

    H3 scans hundreds of subsets against the same 30 (split, draw) r1 fits; without this cache
    ``subset_point`` would re-read and re-decompress each fit's test probabilities per subset.
    """
    cache: DrawCache = {}
    for s in range(N_SPLITS):
        perm = canonical_permutation(config, s, canonical_names)
        draws = []
        for d in range(N_DRAWS):
            rec = read_run_record(
                allocation_dir(ds_paths[s], "r1", d),
                splits=("test",),
                array_fields=("labels", "probabilities"),
            )
            if rec is None:
                raise RuntimeError(f"Missing r1 run record at split {s}, draw {d}")
            test = rec["splits"]["test"]
            draws.append(
                (np.asarray(test["labels"]), np.asarray(test["probabilities"]))
            )
        cache[s] = (perm, draws)
    return cache


def _macro_ba(
    ctx: BootstrapContext, labels: np.ndarray, preds: np.ndarray, k: int
) -> float:
    """Observed (unbootstrapped) patient-macro balanced accuracy, in percent."""
    return float(_patient_macro_recalls(ctx, labels, preds, k)[:, 0].mean()) * 100.0


def _subset_fit_point(
    ctx: BootstrapContext,
    labels_local: np.ndarray,
    probs_full: np.ndarray,
    local_cols: np.ndarray,
    label_to_subset: np.ndarray,
    uniform: np.ndarray,
    shares_100: np.ndarray,
    k: int,
) -> tuple[float, float, np.ndarray]:
    """One draw's (rho=1 restricted BA, rho=100 injected BA, margins) for one subset."""
    probs = renormalize_subset(probs_full, local_cols)
    labels_full = label_to_subset[labels_local]
    in_subset = labels_full >= 0

    ba_r1 = _macro_ba(ctx, labels_full, injected_preds(probs, uniform), k)
    ba_inj = _macro_ba(ctx, labels_full, injected_preds(probs, shares_100), k)

    log_probs = np.log(np.maximum(probs[in_subset], _PROBABILITY_FLOOR))
    true_idx = labels_full[in_subset]
    rows = np.arange(len(true_idx))
    true_logp = log_probs[rows, true_idx]
    masked = log_probs.copy()
    masked[rows, true_idx] = -np.inf
    return ba_r1, ba_inj, true_logp - masked.max(axis=1)


def _subset_split(
    perm_draws: tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]],
    ctx: BootstrapContext,
    subset_idx: np.ndarray,
    n_classes: int,
    uniform: np.ndarray,
    shares_100: np.ndarray,
    k: int,
) -> tuple[list[float], list[float], list[np.ndarray]]:
    """One split's every draw, for one subset and rank assignment."""
    perm, draws = perm_draws
    local_cols = perm[subset_idx]
    label_to_subset = np.full(n_classes, -1, dtype=np.int64)
    label_to_subset[local_cols] = np.arange(k)
    ba_r1, ba_inj, margins = [], [], []
    for labels_local, probs_full in draws:
        r1_ba, inj_ba, m = _subset_fit_point(
            ctx,
            labels_local,
            probs_full,
            local_cols,
            label_to_subset,
            uniform,
            shares_100,
            k,
        )
        ba_r1.append(r1_ba)
        ba_inj.append(inj_ba)
        margins.append(m)
    return ba_r1, ba_inj, margins


def subset_point(
    r1_cache: DrawCache,
    ctxs: dict[int, BootstrapContext],
    canonical_names: list[str],
    subset_names: list[str],
    rank_perm: np.ndarray,
) -> tuple[float, float, float]:
    """Pooled (rho=1 restricted BA, D_sim, median margin) for one subset and rank assignment."""
    k = SUBSET_K
    shares_100 = rho_shares(k, 100.0)[rank_perm]
    uniform = np.full(k, 1.0 / k)
    subset_idx = np.array([canonical_names.index(n) for n in subset_names])
    ba_r1, ba_inj, margins = [], [], []
    for s in range(N_SPLITS):
        r1s, injs, ms = _subset_split(
            r1_cache[s],
            ctxs[s],
            subset_idx,
            len(canonical_names),
            uniform,
            shares_100,
            k,
        )
        ba_r1 += r1s
        ba_inj += injs
        margins += ms
    r1_point = float(np.mean(ba_r1))
    return (
        r1_point,
        r1_point - float(np.mean(ba_inj)),
        float(np.median(np.concatenate(margins))),
    )
