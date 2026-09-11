"""Bootstrap-context recall loading shared by the grid refit and the new fits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.query import read_run_record

from breadth import BOOTSTRAP_SEED, N_REPLICATES, draw_dir, exp2_split_paths
from breadth.analyze.secondary import _patient_macro_recalls

from sites import N_DRAWS, N_SPLITS, allocation_dir

__all__ = [
    "PathsBySplit",
    "contexts",
    "ctx_list",
    "grid_dirs",
    "allocation_dirs",
    "allocation_distribution",
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
    result_dir: Path, class_names: list[str], ctx: BootstrapContext
) -> np.ndarray:
    """Per-class patient-macro recall distributions of one stored fit."""
    rec = read_run_record(
        result_dir, splits=("test",), array_fields=("labels", "preds")
    )
    if rec is None or "test" not in rec.get("splits", {}):
        raise RuntimeError(f"Missing run record at {result_dir}")
    test = rec["splits"]["test"]
    labels, preds = np.asarray(test["labels"]), np.asarray(test["preds"])
    return _patient_macro_recalls(ctx, labels, preds, len(class_names))


def allocation_distribution(
    dirs: list[Path],
    class_names: list[str],
    ctxs: list[BootstrapContext],
    class_idx: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Pooled (n_replicates,) accuracy (%) over ``class_idx`` classes, and its dispersion."""
    per_fit = np.stack(
        [_recall_matrix(d, class_names, ctx) for d, ctx in zip(dirs, ctxs)]
    )
    per_fit_class_mean = per_fit[:, class_idx, :].mean(axis=1) * 100.0  # (F, R)
    pooled = per_fit_class_mean.mean(axis=0)
    dispersion = float(np.std(per_fit_class_mean[:, 0]))
    return pooled, dispersion
