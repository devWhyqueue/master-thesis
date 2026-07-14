from __future__ import annotations

import hashlib
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


def cap_feasible_shared_total(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    seed: int,
) -> int:
    """Find the largest controlled total that satisfies the actual unit caps."""
    supports = class_support_counts(train_df, is_mil)
    available = [supports[name] for name in classes]
    selector = select_slides_round_robin if is_mil else select_patches_round_robin
    for total in range(
        max_shared_total(available, min_support), len(classes) * min_support - 1, -1
    ):
        allocations = [
            allocate_counts(
                available,
                total,
                effective_rho(available, rho, min_support, total),
                min_support,
            )
            for rho in (1.0, 10.0, 100.0)
        ]
        if _cap_feasible(
            train_df, classes, allocations, selector, is_mil, seed, designate_patch_pool
        ):
            return total
    raise ValueError(
        "No shared total satisfies the independent-support and contribution caps"
    )


def _cap_feasible(
    train_df: pd.DataFrame,
    classes: list[str],
    allocations: list[list[int]],
    selector: Callable[..., pd.DataFrame],
    is_mil: bool,
    seed: int,
    designate: Callable[..., pd.DataFrame],
) -> bool:
    """Probe every condition allocation on its designated fixed patch pool."""
    try:
        pools = (
            {
                name: designate(
                    cast(pd.DataFrame, train_df[train_df["cancer_type"] == name]),
                    min(counts[index] for counts in allocations),
                    class_construction_seed(seed, name),
                )
                for index, name in enumerate(classes)
            }
            if not is_mil
            else {}
        )
        if any(
            pool["slide_id"].nunique() > min(counts[index] for counts in allocations)
            for index, pool in enumerate(pools.values())
        ):
            return False
        for counts in allocations:
            for index, name in enumerate(classes):
                selector(
                    pools.get(name, train_df[train_df["cancer_type"] == name]),
                    counts[index],
                    class_construction_seed(seed, name),
                )
    except ValueError:
        return False
    return True
