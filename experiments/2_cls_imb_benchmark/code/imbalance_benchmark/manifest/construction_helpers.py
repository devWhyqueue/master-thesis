from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, cast

from pathlib import Path

import pandas as pd

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.construction import (
    allocate_counts,
    build_manifest_hash,
    effective_rho,
)
from imbalance_benchmark.manifest.sampling.patch_pool import designate_patch_pool
from imbalance_benchmark.manifest.statistics import (
    natural_contribution_stats,
    support_statistics,
)

CONDITION_RHOS = {"balanced": 1.0, "moderate": 10.0, "severe": 100.0}


def apply_class_exclusions(
    df: pd.DataFrame, excluded_classes: list[str]
) -> pd.DataFrame:
    """Drop configured target classes from the eligible pool before splitting.

    Applied once, before ``split_cases``, so an excluded class never enters
    any partition or downstream construction rather than being filtered out
    piecemeal by later stages.
    """
    if not excluded_classes:
        return df
    return cast(
        pd.DataFrame, df[~df["cancer_type"].isin(excluded_classes)]
    ).reset_index(drop=True)


def condition_metadata(
    path: Path,
    condition: pd.DataFrame,
    statistics: dict[str, Any],
    primary: dict[str, Any],
    contributions: dict[str, Any],
    constraints: tuple[str | None, str | None],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Return frozen metadata for one written controlled manifest."""
    return {
        "path": str(path),
        "sha256": compute_sha256(path),
        "requested_rho": CONDITION_RHOS.get(str(spec["name"]), 1.0),
        "achieved_rho": primary["achieved_rho"],
        "normalized_entropy": primary["normalized_entropy"],
        "allocated_counts": primary["counts"],
        "support_statistics": statistics,
        "manifest_hash": build_manifest_hash(condition),
        "contribution_stats": contributions,
        "construction_seed": spec["seed"],
        "evidence_pool_hash": spec["pool_hash"],
        "limiting_class": constraints[0],
        "binding_independent_support_constraint": constraints[1],
    }


def class_support_counts(train_df: pd.DataFrame, is_mil: bool) -> dict[str, int]:
    """Count allocation units: slides for MIL and patches otherwise."""
    if is_mil:
        return train_df.groupby("cancer_type")["slide_id"].nunique().to_dict()
    return train_df["cancer_type"].value_counts().to_dict()


def class_construction_seed(seed: int, class_name: str) -> int:
    """Derive a class-identity seed independent of its assigned tail rank."""
    digest = hashlib.sha256(f"{seed}:definitive:{class_name}".encode()).hexdigest()
    return int(digest[:8], 16)


def write_natural_condition(
    train_df: pd.DataFrame, data_dir: Path, is_mil: bool
) -> dict[str, object]:
    """Write the descriptive full-training-set anchor outside controlled estimands."""
    path = data_dir / "manifest_natural.csv"
    train_df.to_csv(path, index=False)
    statistics = support_statistics(train_df)
    primary = statistics["slide" if is_mil else "patch"]
    return {
        "path": str(path),
        "sha256": compute_sha256(path),
        "note": "descriptive anchor; excluded from imbalance deficit/recovery estimands",
        "allocated_counts": primary["counts"],
        "achieved_rho": primary["achieved_rho"],
        "normalized_entropy": primary["normalized_entropy"],
        "support_statistics": statistics,
        "contribution_stats": natural_contribution_stats(train_df, is_mil),
    }


def assignment_allocations(
    train_df: pd.DataFrame,
    assignments: Mapping[str, list[str]],
    total: int,
    minimum: int,
    is_mil: bool = False,
    condition_names: tuple[str, ...] = tuple(CONDITION_RHOS),
    supports: Mapping[str, int] | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    """Allocate every condition for every locked semantic-class assignment."""
    support_counts = supports or class_support_counts(train_df, is_mil)
    return {
        assignment: {
            condition: dict(
                zip(
                    order,
                    allocate_counts(
                        [support_counts[name] for name in order],
                        total,
                        effective_rho(
                            [support_counts[name] for name in order],
                            CONDITION_RHOS[condition],
                            minimum,
                            total,
                        ),
                        minimum,
                    ),
                    strict=True,
                )
            )
            for condition in condition_names
        }
        for assignment, order in assignments.items()
    }


def required_counts_by_class(
    allocations: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, set[int]]:
    """Every allocated count observed per class, across conditions and assignments."""
    required: dict[str, set[int]] = {}
    for condition_sets in allocations.values():
        for counts in condition_sets.values():
            for class_name, count in counts.items():
                required.setdefault(class_name, set()).add(count)
    return required


def designate_shared_patch_pools(
    train_df: pd.DataFrame,
    allocations: Mapping[str, Mapping[str, Mapping[str, int]]],
    independent_floor: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Designate one per-class pool that every locked allocation draws from.

    The pool is sized to the largest count a class is ever assigned; smaller
    counts (e.g. a tail condition's) draw from it without needing to exhaust
    it, so ``max_pool_units`` must not be clipped to the smallest count.
    """
    required = required_counts_by_class(allocations)
    return {
        class_name: designate_patch_pool(
            cast(pd.DataFrame, train_df[train_df["cancer_type"] == class_name]),
            independent_floor,
            class_construction_seed(seed, class_name),
            max(counts),
            max_pool_units=max(independent_floor, max(counts)),
            required_counts=tuple(sorted(counts)),
        )
        for class_name, counts in required.items()
    }


def _retains_fixed_pool(selected: pd.DataFrame, pool: pd.DataFrame) -> bool:
    """Whether a patch condition retains every designated patient and slide."""
    return set(pool["case_id"]).issubset(selected["case_id"]) and set(
        pool["slide_id"]
    ).issubset(selected["slide_id"])
