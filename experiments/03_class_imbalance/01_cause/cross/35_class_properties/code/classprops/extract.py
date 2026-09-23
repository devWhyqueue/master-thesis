"""Extract stage (0 new fits): every (pool, rho) pool's observations and this dataset's
cross-fitted covariates, written to ``data/extract.npz`` for this dataset's own ``analyze`` and
for the peer dataset's (via ``slurm.peer_outputs``, exp-33's idiom).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from imbalance_benchmark.common import ensure_dirs, write_json

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names

from spectrum import baseline_config

from classprops import (
    POOL_KINDS,
    RATIOS,
    arm_plan,
    sensitivity_c_arm_plan,
    total_arm_plan,
)
from classprops.covariates import (
    cross_fitted_headroom,
    cross_fitted_margin,
    shared_draw_weight,
)
from classprops.pool import (
    FIT_SPLIT,
    Observations,
    build_pool,
    class_recall_stack,
    pooled_ba,
    read_r1_stack,
)

__all__ = ["run_extract"]


def _pool_arrays(obs: Observations, prefix: str) -> dict[str, np.ndarray]:
    """One pool's observation arrays, namespaced by ``{pool}_{rho}``."""
    return {
        f"{prefix}_delta": obs.delta,
        f"{prefix}_z": obs.z,
        f"{prefix}_split_idx": obs.split_idx,
        f"{prefix}_class_idx": obs.class_idx,
        f"{prefix}_weight": obs.draw_weight,
    }


def _sanity(
    config: dict[str, Any],
    names: list[str],
    plan: list[tuple[str, str, str]],
    r1_stack: np.ndarray,
    w_draw: np.ndarray,
) -> dict[str, Any]:
    """Own-computed r1/r100 pooled BA, for the analyze-stage published-value cross-check."""
    ba_by_arm = {}
    for arm, source_key, _ in plan:
        stack = class_recall_stack(baseline_config(config, source_key), names, arm)
        ba_by_arm[arm] = float(pooled_ba(stack, w_draw)[0])
    return {"ba_r1": float(pooled_ba(r1_stack, w_draw)[0]), "ba_by_arm_r100": ba_by_arm}


def _all_pool_arrays(
    config: dict[str, Any],
    names: list[str],
    dataset: str,
    r1_stack: np.ndarray,
    w_draw: np.ndarray,
) -> dict[str, np.ndarray]:
    """Every (pool, rho) pool's arrays, plus TCGA-UT's random-orders-only sensitivity pool."""
    arrays: dict[str, np.ndarray] = {}
    for pool in POOL_KINDS:
        for rho in RATIOS:
            plan = arm_plan(dataset, pool, rho)
            obs = build_pool(config, names, dataset, plan, r1_stack, w_draw)
            arrays.update(_pool_arrays(obs, f"{pool}_{rho}"))
    if dataset == "tcga_ut":
        sens_plan = sensitivity_c_arm_plan(dataset, 100)
        sens_obs = build_pool(config, names, dataset, sens_plan, r1_stack, w_draw)
        arrays.update(_pool_arrays(sens_obs, "sens_c_100"))
    return arrays


def _write_extract(
    config: dict[str, Any],
    dataset: str,
    names: list[str],
    arrays: dict[str, np.ndarray],
    r1_stack: np.ndarray,
    w_draw: np.ndarray,
) -> Path:
    """Write the pooled observation arrays and this dataset's sanity-checkable metadata."""
    data_dir = ensure_dirs(config)["data"]
    np.savez_compressed(data_dir / "extract.npz", **cast(dict[str, Any], arrays))
    write_json(
        data_dir / "extract_meta.json",
        {
            "dataset": dataset,
            "names": names,
            "n_classes": len(names),
            "sanity": _sanity(
                config, names, total_arm_plan(dataset, 100), r1_stack, w_draw
            ),
        },
    )
    return data_dir / "extract.npz"


def run_extract(config: dict[str, Any]) -> Path:
    """Build every pool's observations and this dataset's cross-fitted headroom/margin."""
    dataset = config["dataset"]["name"]
    names = canonical_class_names(config)
    r1_stack = read_r1_stack(config, names)
    n_replicates = r1_stack.shape[-1]
    w_draw = shared_draw_weight(FIT_SPLIT, n_replicates, BOOTSTRAP_SEED)

    arrays: dict[str, np.ndarray] = {
        "h": cross_fitted_headroom(r1_stack, w_draw, names),
        "m": cross_fitted_margin(config, names, n_replicates),
    }
    arrays.update(_all_pool_arrays(config, names, dataset, r1_stack, w_draw))
    return _write_extract(config, dataset, names, arrays, r1_stack, w_draw)
