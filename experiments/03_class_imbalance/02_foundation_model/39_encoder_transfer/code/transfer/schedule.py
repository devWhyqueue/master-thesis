"""Phase-01 schedule: patient cohort, class permutation, and exact training-patch identities for
exp-39's draws 10-19, generated once per dataset/split without loading either encoder's features
(``configs/protocol_lock.json``). Each encoder's later fit consumes this same schedule
by matching patch identity, not by re-deriving row order from its own feature cache.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.manifest.statistics import achieved_rho

from breadth import exp2_split_paths

from centre.cohort import patient_rows

from prevalence.fit import _patient_counts, _shard_context, class_counts

from transfer import FIT_SOURCE

__all__ = ["load_train_identity", "draw_schedule"]

_ID_COLS = ("case_id", "slide_id", "patch_id")


def load_train_identity(
    config: dict[str, Any], split_idx: int
) -> tuple[pd.DataFrame, list[str]]:
    """Train-split identity rows and class names from exp-2's manifest; no feature columns touched."""
    exp2_p = exp2_split_paths(config, split_idx)
    m_file = exp2_p["data"] / "manifest.csv"
    classes = list(load_freeze_meta(exp2_p)["class_names"])
    train_df = pd.read_csv(m_file).query("split == 'train'").reset_index(drop=True)
    return train_df, classes


def _identity_rows(train_df: pd.DataFrame, rows: list[int]) -> list[list[str]]:
    """(case_id, slide_id, patch_id) triplets for a set of manifest row indices."""
    cols = [c for c in _ID_COLS if c in train_df.columns]
    if not cols:
        raise RuntimeError("Manifest carries no patch identity columns")
    return train_df.loc[rows, cols].astype(str).values.tolist()


def _arm_rows(
    train_df: pd.DataFrame,
    names: list[str],
    patients: list[list[str]],
    counts: list[int],
) -> list[int]:
    """Row indices of one arm's realized allocation (mirrors ``prevalence.fit._arm_rows``, identity only)."""
    rows: list[int] = []
    for ci, name in enumerate(names):
        class_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == name])
        for patient, m in zip(
            patients[ci], _patient_counts(counts[ci], len(patients[ci]))
        ):
            if m:
                rows.extend(patient_rows(class_df, [patient], m))
    return rows


def _arm_entry(
    train_df: pd.DataFrame,
    names: list[str],
    data_arm: str,
    prior_arm: str | None,
    counts_by_arm: dict[str, list[int]],
    rows_by_arm: dict[str, list[int]],
) -> dict[str, Any]:
    """One arm's realized counts, ratio, row count, and patch identity."""
    counts = dict(zip(names, (int(c) for c in counts_by_arm[data_arm])))
    entry: dict[str, Any] = {
        "data_arm": data_arm,
        "prior_arm": prior_arm,
        "class_counts": counts,
        "realized_rho": achieved_rho(counts),
        "n_rows": len(rows_by_arm[data_arm]),
        "patch_identity": _identity_rows(train_df, rows_by_arm[data_arm]),
    }
    if prior_arm is not None:
        entry["prior_counts"] = dict(
            zip(names, (int(c) for c in counts_by_arm[prior_arm]))
        )
    return entry


def draw_schedule(
    train_df: pd.DataFrame, names: list[str], split_idx: int, draw_idx: int, g: int
) -> dict[str, Any]:
    """One (split, draw) cell's frozen cohort, permutation, and every arm's patch identities."""
    shard = _shard_context(train_df, names, split_idx, draw_idx, g)
    data_arms = sorted({d for d, _ in FIT_SOURCE.values()})
    counts_by_arm = {
        a: class_counts(a, shard.perm, shard.available, shard.pool_counts, shard.g)
        for a in data_arms
    }
    rows_by_arm = {
        a: _arm_rows(train_df, names, shard.patients, counts_by_arm[a])
        for a in data_arms
    }
    arms = {
        arm: _arm_entry(
            train_df, names, data_arm, prior_arm, counts_by_arm, rows_by_arm
        )
        for arm, (data_arm, prior_arm) in FIT_SOURCE.items()
    }
    return {
        "split": split_idx,
        "draw": draw_idx,
        "class_permutation": [names[i] for i in shard.perm],
        "patients": dict(zip(names, shard.patients)),
        "arms": arms,
    }
