from __future__ import annotations

import logging
from typing import cast
import numpy as np
import pandas as pd
from imbalance_benchmark.common import compute_data_hash
from imbalance_benchmark.datasets.bracs import LABELS as BRACS_LABELS
from imbalance_benchmark.manifest.construction_sampling import (
    select_patches_round_robin,
    select_slides_round_robin,
)

logger = logging.getLogger(__name__)

__all__ = [
    "split_cases",
    "validate_split_leakage",
    "allocate_counts",
    "effective_rho",
    "max_shared_total",
    "select_patches_round_robin",
    "select_slides_round_robin",
    "build_manifest_hash",
    "patient_equals_slide",
    "locked_class_names",
]


def locked_class_names(df: pd.DataFrame) -> list[str]:
    """Return the semantic native order and require every target in each split."""
    classes = sorted(df["cancer_type"].astype(str).unique().tolist())
    if set(classes) == set(BRACS_LABELS):
        classes = list(BRACS_LABELS)
    elif classes and all(name.startswith("ISUP") for name in classes):
        classes = sorted(classes, key=lambda name: int(name.removeprefix("ISUP")))
    else:
        counts = df.loc[df["split"] == "train", "cancer_type"].value_counts()
        classes = sorted(classes, key=lambda name: (-int(counts.get(name, 0)), name))
    expected = set(classes)
    for split, frame in df.groupby("split", sort=False):
        missing = sorted(expected - set(frame["cancer_type"].astype(str)))
        if missing:
            raise ValueError(f"Split '{split}' lacks locked target classes: {missing}")
    return [str(name) for name in classes]


def split_cases(
    df: pd.DataFrame, seed: int, val_frac: float = 0.15, test_frac: float = 0.15
) -> pd.DataFrame:
    """Stratify patients disjointly into train, validation, and test splits."""
    case_labels = df.groupby("case_id")["cancer_type"].first().reset_index()
    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for _, group in case_labels.groupby("cancer_type"):
        cases = group["case_id"].to_numpy().astype(str)
        rng.shuffle(cases)
        n = len(cases)
        if n <= 2:
            for idx, c in enumerate(cases):
                assignments[c] = "train" if idx == 0 else "test"
        else:
            nt, nv = max(1, int(round(n * test_frac))), max(1, int(round(n * val_frac)))
            if nt + nv >= n:
                nt, nv = 1, 1
            for c in cases[:nt]:
                assignments[c] = "test"
            for c in cases[nt : nt + nv]:
                assignments[c] = "validation"
            for c in cases[nt + nv :]:
                assignments[c] = "train"
    df_splits = df.copy()
    df_splits["split"] = df_splits["case_id"].astype(str).map(assignments.get)
    return df_splits


def validate_split_leakage(df: pd.DataFrame) -> None:
    """Raise if any case/patient is assigned to more than one split partition."""
    counts = df.groupby("case_id")["split"].nunique()
    leaking = cast(pd.Series, counts[counts > 1])
    if not leaking.empty:
        raise RuntimeError(
            f"Patient-disjoint split violated for cases: {leaking.index.tolist()[:10]}"
        )


def patient_equals_slide(df: pd.DataFrame) -> bool:
    """Return whether every case contributes at most one slide (e.g. CAMELYON16/PANDA)."""
    return bool(df.groupby("case_id")["slide_id"].nunique().max() <= 1)


def build_manifest_hash(manifest_df: pd.DataFrame) -> str:
    """Create a stable hash of the identifiers defining a frozen manifest."""
    columns = ["case_id", "slide_id", "cancer_type", "split"]
    records = (
        cast(pd.DataFrame, manifest_df[columns])
        .sort_values(by=["split", "cancer_type", "slide_id"])
        .to_dict("records")
    )
    return compute_data_hash(records)


def _adjust_alloc(
    allocated: list[int],
    available: list[int],
    target: list[float],
    diff: int,
    min_support: int,
) -> None:
    """Adjust counts up or down to match target exactly."""
    k = len(allocated)
    if diff > 0:
        while diff > 0:
            cands = [i for i in range(k) if allocated[i] < available[i]]
            if not cands:
                break
            cands.sort(key=lambda idx: target[idx] - allocated[idx], reverse=True)
            allocated[cands[0]] += 1
            diff -= 1
    elif diff < 0:
        while diff < 0:
            cands = [i for i in range(k) if allocated[i] > min_support]
            if not cands:
                break
            cands.sort(key=lambda idx: allocated[idx] - target[idx], reverse=True)
            allocated[cands[0]] -= 1
            diff += 1


def allocate_counts(
    available: list[int], total_t: int, rho: float, min_support: int
) -> list[int]:
    """Perform constrained integer allocation for class counts under exponential formula."""
    k = len(available)
    if k == 0:
        return []
    if k == 1:
        return [min(total_t, available[0])]
    w = [rho ** (-i / (k - 1)) for i in range(k)]
    sum_w = sum(w)
    target = [total_t * val / sum_w for val in w]
    allocated = [
        int(np.clip(round(t), min_support, avail))
        for t, avail in zip(target, available)
    ]
    _adjust_alloc(allocated, available, target, total_t - sum(allocated), min_support)
    return allocated


def max_shared_total(
    available: list[int], min_support: int, rhos: tuple[float, ...] = (1.0, 10.0, 100.0)
) -> int:
    """Return the maximum total of an approximately balanced controlled condition.

    Severity is deliberately excluded: requested ratios are lowered only
    after this shared total has been fixed. ``rhos`` is retained for
    compatibility with older callers.
    """
    if not available or min(available) < min_support:
        raise ValueError("No shared total satisfies the independent-support floor")
    del rhos
    minimum = min(available)
    return len(available) * minimum + sum(capacity > minimum for capacity in available)


def _allocation_is_feasible(
    available: list[int], total_t: int, rho: float, min_support: int
) -> bool:
    """Whether an un-clipped integer profile can realize ``rho`` at ``total_t``."""
    k = len(available)
    if k == 1:
        return min_support <= total_t <= available[0]
    weights = np.asarray([rho ** (-i / (k - 1)) for i in range(k)], dtype=float)
    target = total_t * weights / weights.sum()
    if any(
        value < min_support - 0.5 or value >= capacity + 1.0
        for value, capacity in zip(target, available, strict=True)
    ):
        return False
    allocated = [
        int(np.clip(round(value), min_support, capacity))
        for value, capacity in zip(target, available, strict=True)
    ]
    _adjust_alloc(
        allocated, available, target.tolist(), total_t - sum(allocated), min_support
    )
    return sum(allocated) == total_t and all(
        min_support <= count <= cap
        for count, cap in zip(allocated, available, strict=True)
    )


def effective_rho(
    available: list[int], rho: float, min_support: int, total_t: int | None = None
) -> float:
    """Lower a requested ratio to the largest allocation-feasible value.

    Feasibility is evaluated using the complete exponential allocation, every
    class-specific availability cap, and the requested shared total.  This
    avoids the invalid head-only shortcut that can reject attainable designs.
    """
    if not available or min_support < 1:
        raise ValueError("Support and floor must be positive")
    if total_t is None:
        return min(rho, available[0] / min_support)
    if total_t < len(available) * min_support:
        raise ValueError("Shared total cannot satisfy the independent-support floor")

    if _allocation_is_feasible(available, total_t, rho, min_support):
        return rho
    return _largest_feasible_rho(available, total_t, rho, min_support)


def _largest_feasible_rho(
    available: list[int], total_t: int, rho: float, min_support: int
) -> float:
    """Binary-search the largest realizable severity at a frozen shared total."""
    low, high = 1.0, rho
    if not _allocation_is_feasible(available, total_t, low, min_support):
        raise ValueError(
            "No shared total satisfies all availability and support constraints"
        )
    for _ in range(48):
        mid = (low + high) / 2.0
        if _allocation_is_feasible(available, total_t, mid, min_support):
            low = mid
        else:
            high = mid
    return low
