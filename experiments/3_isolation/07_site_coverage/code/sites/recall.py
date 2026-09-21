"""Bootstrap-context recall loading shared by the grid refit and the new fits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.query import read_run_record

from breadth import BOOTSTRAP_SEED, N_REPLICATES, draw_dir, exp2_split_paths
from breadth.analyze.secondary import _patient_macro_recalls
from breadth.analyze.canonical import canonical_permutation

from sites import N_DRAWS, N_SPLITS, allocation_dir

__all__ = [
    "PathsBySplit",
    "contexts",
    "ctx_list",
    "grid_dirs",
    "allocation_dirs",
    "allocation_distribution",
    "perm_list",
]

PathsBySplit = dict[int, dict[str, Path]]


def contexts(config: dict[str, Any]) -> dict[int, BootstrapContext]:
    """Build one BootstrapContext per split, matching exp-5/exp-6's own."""
    return {
        s: BootstrapContext(
            exp2_split_paths(config, s),
            is_mil=False,
            n_replicates=N_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        for s in range(N_SPLITS)
    }


def ctx_list(ctxs: dict[int, BootstrapContext]) -> list[BootstrapContext]:
    """Flatten per-split contexts to one entry per (split, draw) fit."""
    return [ctxs[s] for s in range(N_SPLITS) for _ in range(N_DRAWS)]


def grid_dirs(paths5: PathsBySplit, g: int, m: int) -> list[Path]:
    """Every stored exp-5 draw directory for one grid cell."""
    return [
        draw_dir(paths5[s], g, m, d) for s in range(N_SPLITS) for d in range(N_DRAWS)
    ]


def allocation_dirs(paths7: PathsBySplit, allocation: str) -> list[Path]:
    """Every stored exp-7 draw directory for one allocation."""
    return [
        allocation_dir(paths7[s], allocation, d)
        for s in range(N_SPLITS)
        for d in range(N_DRAWS)
    ]


def _recall_matrix(
    result_dir: Path, ctx: BootstrapContext, n_classes: int, perm: np.ndarray
) -> np.ndarray:
    """Per-class patient-macro recall distributions of one stored fit.

    The stored fit's own split may have frozen its classes in a different
    order than ``perm``'s target order, so the returned rows are reordered
    (``perm``, from :func:`breadth.analyze.canonical.canonical_permutation`) before
    return -- every caller can then treat row ``c`` as meaning the same class
    regardless of which split a given fit came from.
    """
    rec = read_run_record(
        result_dir, splits=("test",), array_fields=("labels", "preds")
    )
    if rec is None or "test" not in rec.get("splits", {}):
        raise RuntimeError(f"Missing run record at {result_dir}")
    test = rec["splits"]["test"]
    labels, preds = np.asarray(test["labels"]), np.asarray(test["preds"])
    return _patient_macro_recalls(ctx, labels, preds, n_classes)[perm]


def perm_list(config: dict[str, Any], canonical_names: list[str]) -> list[np.ndarray]:
    """One canonicalizing permutation per fit, in :func:`ctx_list`/dir-list order."""
    per_split = [
        canonical_permutation(config, s, canonical_names) for s in range(N_SPLITS)
    ]
    return [per_split[s] for s in range(N_SPLITS) for _ in range(N_DRAWS)]


def allocation_distribution(
    dirs: list[Path],
    ctxs: list[BootstrapContext],
    perms: list[np.ndarray],
    n_classes: int,
    class_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (R,) and per-fit observed (F,) accuracy (%) over the given classes."""
    per_fit = np.stack(
        [_recall_matrix(d, ctx, n_classes, p) for d, ctx, p in zip(dirs, ctxs, perms)]
    )
    per_fit_class_mean = per_fit[:, class_idx, :].mean(axis=1) * 100.0  # (F, R)
    return per_fit_class_mean.mean(axis=0), per_fit_class_mean[:, 0]
