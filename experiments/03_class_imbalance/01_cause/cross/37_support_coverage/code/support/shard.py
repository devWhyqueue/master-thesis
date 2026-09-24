"""Per-(split, draw) class pools and the B/random/coverage/redundant training data built from
them. Reuses exp-25's own patient draw, allocator and row loader, and exp-36's own dataset_shard
(TCGA-UT's G=10 nested prefix / BRACS's native G=10) unchanged.

S is capped at the B pool's own size (320 = G x BALANCED) so ``S ⊂ B`` holds for every class, not
only the S100 tail (PLAN.md "S ⊂ B always, so B is identical across arms"): exp-27/28's own r100
allocator gives 1-2 head classes far more than 320 patches (the fixed total budget concentrated
onto a couple of ranks), which the frozen 32-row-per-patient B pool cannot supply. Capping means
this experiment's own "random" S arm is not bit-identical to exp-36's S100 arm on those 1-2 head
classes; every other (thinned) class is unaffected, since its own r100 count already undershoots
320.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from breadth.sampling import load_features_for_df
from centre.cohort import patient_rows
from prevalence.fit import _Shard, _patient_counts, class_counts

from joint.fitting.shard import TrainingData

from support import BALANCED, SEVERITY
from support.coverage import patient_balanced_mean
from support.selection import greedy_quota_coverage, prefix_indices, redundant_pick

__all__ = ["ClassPool", "class_pools", "arm_training_data", "b_training_data"]


@dataclass(frozen=True)
class ClassPool:
    """One class's frozen B pool: features, each row's patient index, and its S100 quota."""

    class_index: int
    x: np.ndarray  # (n_pool, d), n_pool == g * BALANCED
    patient_idx: np.ndarray  # (n_pool,) int, 0..g-1
    quota: list[int]  # per-patient S-arm pick count, sum(quota) == s_count
    s_count: int  # this class's r100 total, capped at n_pool
    thinned: (
        bool  # s_count < n_pool: this class's r100 count already undershoots the B pool
    )


def _class_b_rows(
    train_df: pd.DataFrame, name: str, patients: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """This class's frozen B pool (BALANCED rows/patient) and each row's patient index."""
    class_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == name])
    rows: list[int] = []
    patient_idx: list[int] = []
    for pi, patient in enumerate(patients):
        prows = patient_rows(class_df, [patient], BALANCED)
        rows.extend(prows)
        patient_idx.extend([pi] * len(prows))
    x = load_features_for_df(train_df.loc[rows]).astype(np.float64)
    return x, np.asarray(patient_idx, dtype=np.int64)


def class_pools(shard: _Shard) -> list[ClassPool]:
    """Every class's B pool, S100-derived count (capped) and per-patient quota, for one shard."""
    counts_raw = class_counts(
        f"r{SEVERITY}", shard.perm, shard.available, shard.pool_counts, shard.g
    )
    pools = []
    for ci, name in enumerate(shard.names):
        x, patient_idx = _class_b_rows(shard.train_df, name, shard.patients[ci])
        n_pool = len(x)
        s_count = min(int(counts_raw[ci]), n_pool)
        quota = _patient_counts(s_count, shard.g)
        pools.append(ClassPool(ci, x, patient_idx, quota, s_count, s_count < n_pool))
    return pools


def _select(pool: ClassPool, arm: str) -> np.ndarray:
    if arm == "random":
        return prefix_indices(pool.patient_idx, pool.quota)
    if arm == "coverage":
        return greedy_quota_coverage(pool.x, pool.patient_idx, pool.quota)
    if arm == "redundant":
        centre = patient_balanced_mean(pool.x, pool.patient_idx)
        return redundant_pick(pool.x, pool.patient_idx, pool.quota, centre)
    raise ValueError(f"Unknown S arm: {arm}")


def arm_training_data(pools: list[ClassPool], arm: str) -> TrainingData:
    """One S arm's training features, integer targets and realized per-class counts."""
    xs, ys, counts = [], [], []
    for pool in pools:
        idx = _select(pool, arm)
        xs.append(pool.x[idx])
        ys.append(np.full(len(idx), pool.class_index, dtype=np.int64))
        counts.append(len(idx))
    return TrainingData(np.concatenate(xs), np.concatenate(ys), counts, None, None)


def b_training_data(pools: list[ClassPool]) -> TrainingData:
    """The B arm's training features, integer targets and per-class counts: the whole pool."""
    xs = [pool.x for pool in pools]
    ys = [np.full(len(pool.x), pool.class_index, dtype=np.int64) for pool in pools]
    counts = [len(pool.x) for pool in pools]
    return TrainingData(np.concatenate(xs), np.concatenate(ys), counts, None, None)
