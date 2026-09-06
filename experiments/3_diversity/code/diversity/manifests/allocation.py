"""Per-allocation narrow/random/wide manifest construction and its invariant checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.manifest.construction_helpers import class_support_counts
from imbalance_benchmark.manifest.freeze import write_condition

from diversity.manifests.constants import ANCHOR_ASSIGNMENT, LEVELS, SOURCE_MANIFEST
from diversity.manifests.pool import SLOT_KEYS, eligible_pool, pool_features, slot_table
from diversity.manifests.selection import _select_narrow, _select_wide

__all__ = ["build_allocation_levels", "assert_invariants"]


def _load_allocation_inputs(
    allocation: str, exp2_data_dir: Path, class_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """Read the source condition, the train split, its slot table, pool, and features."""
    source_df = cast(
        pd.DataFrame, pd.read_csv(exp2_data_dir / SOURCE_MANIFEST[allocation])
    )
    train_df = cast(
        pd.DataFrame,
        pd.read_csv(exp2_data_dir / "manifest.csv").pipe(
            lambda df: df[df["split"] == "train"]
        ),
    )
    slots = slot_table(source_df)
    pool = eligible_pool(train_df, slots)
    features = pool_features(pool, class_names)
    return source_df, train_df, slots, pool, features


def _select_per_slot(
    pool: pd.DataFrame, slots: pd.DataFrame, features: Any
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    """Narrow and wide row selections for every (class, case, slide) slot, plus headroom."""
    feature_index = pool["feature_index"].to_numpy()
    n_by_slot = {
        (row["cancer_type"], row["case_id"], row["slide_id"]): int(row["n"])
        for row in slots.to_dict("records")
    }
    narrow_rows, wide_rows, headroom_rows = [], [], []
    for key, group in pool.groupby(list(SLOT_KEYS), sort=False):
        cls, case, slide = cast(tuple[str, str, str], key)
        n = n_by_slot[(cls, case, slide)]
        positions = group.index.to_numpy()
        feats, fidx = features[positions], feature_index[positions]
        narrow_rows.append(group.iloc[_select_narrow(feats, fidx, n)])
        wide_rows.append(group.iloc[_select_wide(feats, fidx, n)])
        headroom_rows.append(
            {
                "cancer_type": cls,
                "case_id": case,
                "slide_id": slide,
                "n": n,
                "pool_size": len(group),
                "h": len(group) / n if n else float("nan"),
            }
        )
    return narrow_rows, wide_rows, pd.DataFrame(headroom_rows)


def _per_class_rows(rows: pd.DataFrame, class_names: list[str]) -> list[pd.DataFrame]:
    by_class = dict(tuple(rows.groupby("cancer_type")))
    return [
        cast(pd.DataFrame, by_class[name]) for name in class_names if name in by_class
    ]


def _source_condition_meta(
    exp2_freeze: dict[str, Any], allocation: str
) -> dict[str, Any]:
    """The exp-2 frozen condition metadata backing one exp-3 allocation."""
    if allocation == "balanced":
        return exp2_freeze["conditions"]["balanced"]
    anchor = ANCHOR_ASSIGNMENT[allocation]
    return exp2_freeze["assignment_conditions"][anchor][allocation]


def _write_level(
    allocation: str,
    level: str,
    rows_by_class: list[pd.DataFrame],
    pool: pd.DataFrame,
    data_dir: Path,
    exp2_condition: dict[str, Any],
    class_names: list[str],
    train_df: pd.DataFrame,
    min_support: int,
) -> dict[str, Any]:
    """Write one level's manifest and rebuild its frozen condition metadata."""
    allocated = {
        name: exp2_condition["allocated_counts"].get(name, 0) for name in class_names
    }
    available = [
        class_support_counts(train_df, False).get(name, 0) for name in class_names
    ]
    spec = {
        "name": allocation,
        "allocated": allocated,
        "rows": rows_by_class,
        "pool": pool,
        "is_mil": False,
        "seed": exp2_condition.get("construction_seed"),
        "data_dir": data_dir,
        "stem": f"{allocation}_{level}",
        "pool_hash": exp2_condition.get("evidence_pool_hash"),
        "available": available,
        "minimum": min_support,
        "spread_classes": exp2_condition.get("spread_classes"),
        "spread_ratio": exp2_condition.get("spread_ratio"),
        "spread_tail_classes": exp2_condition.get("spread_tail_classes"),
    }
    return write_condition(spec)


def _write_all_levels(
    allocation: str,
    dataframes: dict[str, pd.DataFrame],
    pool: pd.DataFrame,
    exp3_data_dir: Path,
    exp2_freeze: dict[str, Any],
    class_names: list[str],
    train_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    exp2_condition = _source_condition_meta(exp2_freeze, allocation)
    min_support = int(exp2_freeze["min_support"])
    return {
        level: _write_level(
            allocation,
            level,
            _per_class_rows(dataframes[level], class_names),
            pool,
            exp3_data_dir,
            exp2_condition,
            class_names,
            train_df,
            min_support,
        )
        for level in LEVELS
    }


def _concat_or_empty(rows: list[pd.DataFrame], pool: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(rows, ignore_index=True) if rows else pool.iloc[0:0]


def build_allocation_levels(
    allocation: str,
    exp2_data_dir: Path,
    exp3_data_dir: Path,
    exp2_freeze: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    """Build the narrow/random/wide manifests and metadata for one allocation.

    Returns ``{"conditions": {level: metadata}, "dataframes": {level: df},
    "headroom": DataFrame}``. Raises ``RuntimeError`` (never warns) on any
    invariant violation before anything is treated as usable.
    """
    source_df, train_df, slots, pool, features = _load_allocation_inputs(
        allocation, exp2_data_dir, class_names
    )
    narrow_rows, wide_rows, headroom = _select_per_slot(pool, slots, features)
    dataframes = {
        "random": source_df,
        "narrow": _concat_or_empty(narrow_rows, pool),
        "wide": _concat_or_empty(wide_rows, pool),
    }
    conditions = _write_all_levels(
        allocation, dataframes, pool, exp3_data_dir, exp2_freeze, class_names, train_df
    )
    assert_invariants(allocation, dataframes, conditions, headroom)
    return {"conditions": conditions, "dataframes": dataframes, "headroom": headroom}


def _assert_counts_equal(allocation: str, dataframes: dict[str, pd.DataFrame]) -> None:
    counts = {
        level: df.groupby(list(SLOT_KEYS)).size().sort_index()
        for level, df in dataframes.items()
    }
    reference = counts["random"]
    for level, count in counts.items():
        if not count.equals(reference):
            raise RuntimeError(
                f"{allocation}/{level}: per-(class, case, slide) counts differ from 'random'"
            )


def _assert_identity_sets_equal(
    allocation: str, dataframes: dict[str, pd.DataFrame]
) -> None:
    for key in ("case_id", "slide_id"):
        reference_set = set(dataframes["random"][key])
        for level, df in dataframes.items():
            if set(df[key]) != reference_set:
                raise RuntimeError(
                    f"{allocation}/{level}: {key} set differs from 'random'"
                )


def _assert_metadata_equal(
    allocation: str, conditions: dict[str, dict[str, Any]]
) -> None:
    reference_allocated = conditions["random"]["allocated_counts"]
    reference_contrib = conditions["random"]["contribution_stats"]
    for level, meta in conditions.items():
        if meta["allocated_counts"] != reference_allocated:
            raise RuntimeError(
                f"{allocation}/{level}: allocated_counts differ from 'random'"
            )
        if meta["contribution_stats"] != reference_contrib:
            raise RuntimeError(
                f"{allocation}/{level}: contribution_stats differ from 'random'"
            )


def _assert_narrow_wide_differ_where_headroom(
    allocation: str, dataframes: dict[str, pd.DataFrame], headroom: pd.DataFrame
) -> None:
    contested = headroom[headroom["h"] > 1.0]
    if contested.empty:
        return
    narrow_ids = set(dataframes["narrow"]["patch_id"])
    wide_ids = set(dataframes["wide"]["patch_id"])
    if narrow_ids == wide_ids:
        raise RuntimeError(
            f"{allocation}: narrow and wide selections are identical even though "
            f"{len(contested)} slot(s) have headroom h > 1; selection is not doing anything"
        )


def assert_invariants(
    allocation: str,
    dataframes: dict[str, pd.DataFrame],
    conditions: dict[str, dict[str, Any]],
    headroom: pd.DataFrame,
) -> None:
    """Fail loudly (never warn) if the three levels disagree on anything but patch identity."""
    _assert_counts_equal(allocation, dataframes)
    _assert_identity_sets_equal(allocation, dataframes)
    _assert_metadata_equal(allocation, conditions)
    _assert_narrow_wide_differ_where_headroom(allocation, dataframes, headroom)
