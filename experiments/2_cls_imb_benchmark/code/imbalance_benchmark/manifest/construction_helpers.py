from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Callable, cast

from pathlib import Path

import pandas as pd

from imbalance_benchmark.common import compute_data_hash, compute_sha256
from imbalance_benchmark.construction import (
    allocate_counts,
    effective_rho,
    max_shared_total,
)
from imbalance_benchmark.manifest.construction_sampling import (
    designate_patch_pool,
    select_patches_round_robin,
    select_slides_round_robin,
)

CONDITION_RHOS = {"balanced": 1.0, "moderate": 10.0, "severe": 100.0}


def class_support_counts(train_df: pd.DataFrame, is_mil: bool) -> dict[str, int]:
    """Count allocation units: slides for MIL and patches otherwise."""
    if is_mil:
        return train_df.groupby("cancer_type")["slide_id"].nunique().to_dict()
    return train_df["cancer_type"].value_counts().to_dict()


def class_construction_seed(seed: int, class_name: str) -> int:
    """Derive a class-identity seed independent of its assigned tail rank."""
    digest = hashlib.sha256(f"{seed}:definitive:{class_name}".encode()).hexdigest()
    return int(digest[:8], 16)


def evidence_pool_hash(train_df: pd.DataFrame, classes: list[str], is_mil: bool) -> str:
    """Hash the fixed per-class patient/slide evidence pools shared by conditions."""
    columns = ["cancer_type", "case_id", "slide_id"]
    if not is_mil and "patch_id" in train_df:
        columns.append("patch_id")
    pool = pd.DataFrame(train_df.loc[train_df["cancer_type"].isin(classes), columns])
    pool = cast(pd.DataFrame, pool.sort_values(by=columns))
    return compute_data_hash(pool.to_dict("records"))


def write_natural_condition(
    train_df: pd.DataFrame, data_dir: Path
) -> dict[str, object]:
    """Write the descriptive full-training-set anchor outside controlled estimands."""
    path = data_dir / "manifest_natural.csv"
    train_df.to_csv(path, index=False)
    return {
        "path": str(path),
        "sha256": compute_sha256(path),
        "note": "descriptive anchor; excluded from imbalance deficit/recovery estimands",
    }


def assignment_allocations(
    train_df: pd.DataFrame,
    assignments: Mapping[str, list[str]],
    total: int,
    minimum: int,
    condition_names: tuple[str, ...] = tuple(CONDITION_RHOS),
) -> dict[str, dict[str, dict[str, int]]]:
    """Allocate every condition for every locked semantic-class assignment."""
    supports = class_support_counts(train_df, is_mil=False)
    return {
        assignment: {
            condition: dict(
                zip(
                    order,
                    allocate_counts(
                        [supports[name] for name in order],
                        total,
                        effective_rho(
                            [supports[name] for name in order],
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


def designate_shared_patch_pools(
    train_df: pd.DataFrame,
    allocations: Mapping[str, Mapping[str, Mapping[str, int]]],
    independent_floor: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Designate one per-class pool that can realize every locked allocation."""
    maximums: dict[str, int] = {}
    for condition_sets in allocations.values():
        for counts in condition_sets.values():
            for class_name, count in counts.items():
                maximums[class_name] = max(maximums.get(class_name, 0), count)
    return {
        class_name: designate_patch_pool(
            cast(pd.DataFrame, train_df[train_df["cancer_type"] == class_name]),
            independent_floor,
            class_construction_seed(seed, class_name),
            maximum,
        )
        for class_name, maximum in maximums.items()
    }


def cap_feasible_shared_total(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    seed: int,
    independent_floor: int = 10,
    assignments: Mapping[str, list[str]] | None = None,
) -> int:
    """Find the largest controlled total that satisfies the actual unit caps."""
    supports = class_support_counts(train_df, is_mil)
    available = [supports[name] for name in classes]
    selector = select_slides_round_robin if is_mil else select_patches_round_robin
    for total in range(
        max_shared_total(available, min_support), len(classes) * min_support - 1, -1
    ):
        locked_assignments = assignments or {"native": classes}
        if _cap_feasible(
            train_df,
            locked_assignments,
            total,
            min_support,
            selector,
            is_mil,
            seed,
            independent_floor,
        ):
            return total
    raise ValueError(
        "No shared total satisfies the independent-support and contribution caps"
    )


def _cap_feasible(
    train_df: pd.DataFrame,
    assignments: Mapping[str, list[str]],
    total: int,
    minimum: int,
    selector: Callable[..., pd.DataFrame],
    is_mil: bool,
    seed: int,
    independent_floor: int,
) -> bool:
    """Probe every condition allocation on its designated fixed patch pool."""
    try:
        allocations = assignment_allocations(train_df, assignments, total, minimum)
        pools = (
            designate_shared_patch_pools(train_df, allocations, independent_floor, seed)
            if not is_mil
            else {}
        )
        for condition_sets in allocations.values():
            for counts in condition_sets.values():
                for name, count in counts.items():
                    selected = selector(
                        pools.get(
                            name,
                            cast(
                                pd.DataFrame,
                                train_df[train_df["cancer_type"] == name],
                            ),
                        ),
                        count,
                        class_construction_seed(seed, name),
                    )
                    if not is_mil and not _retains_fixed_pool(selected, pools[name]):
                        return False
    except ValueError:
        return False
    return True


def _retains_fixed_pool(selected: pd.DataFrame, pool: pd.DataFrame) -> bool:
    """Whether a patch condition includes every designated patient and slide."""
    return set(pool["case_id"]).issubset(selected["case_id"]) and set(
        pool["slide_id"]
    ).issubset(selected["slide_id"])
