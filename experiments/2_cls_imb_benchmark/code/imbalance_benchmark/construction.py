from __future__ import annotations

import logging
from typing import cast
import numpy as np
import pandas as pd

from imbalance_benchmark.common import compute_data_hash

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
]


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
    """Return the largest total feasible for every requested condition.

    The balanced maximum alone is insufficient: at that total an imbalanced
    allocation can require more examples from its head class than are unique
    in the training pool.  Search only totals that can meet every independent
    support floor and require each constrained allocation to sum exactly.
    """
    if not available or min(available) < min_support:
        raise ValueError("No shared total satisfies the independent-support floor")
    effective_rhos = tuple(effective_rho(available, rho, min_support) for rho in rhos)
    upper = len(available) * min(available)
    for total in range(upper, len(available) * min_support - 1, -1):
        allocations = [
            allocate_counts(available, total, rho, min_support) for rho in effective_rhos
        ]
        retains_requested_skew = all(
            max(counts) / min(counts)
            >= rho
            for rho, counts in zip(effective_rhos, allocations, strict=True)
        )
        if retains_requested_skew and all(sum(counts) == total for counts in allocations):
            return total
    raise ValueError("No shared total is feasible for every requested condition")


def effective_rho(available: list[int], rho: float, min_support: int) -> float:
    """Lower an infeasible requested ratio to the largest floor-compatible ratio.

    The first class is the assigned head.  A head cannot exceed its unique
    support and every tail must retain the independent-support floor.
    """
    if not available or min_support < 1:
        raise ValueError("Support and floor must be positive")
    return min(rho, available[0] / min_support)


def _build_patch_hierarchy(
    df_class: pd.DataFrame, rng: np.random.Generator
) -> tuple[list[str], dict[str, dict[str, list[int]]]]:
    """Build nested randomized dictionary for patch sampling."""
    patients = cast(np.ndarray, df_class["case_id"].unique())
    rng.shuffle(patients)
    h: dict[str, dict[str, list[int]]] = {}
    for pat in patients:
        h[pat] = {}
        pat_df = cast(pd.DataFrame, df_class[df_class["case_id"] == pat])
        slides = cast(np.ndarray, pat_df["slide_id"].unique())
        rng.shuffle(slides)
        for sld in slides:
            pids = cast(np.ndarray, pat_df[pat_df["slide_id"] == sld].index.to_numpy())
            rng.shuffle(pids)
            h[pat][sld] = list(pids)
    return list(patients), h


def _loop_patches(
    patients: list[str], h: dict, max_p: int, max_s: int, n: int
) -> tuple[list[int], dict, dict]:
    """Execute loop for round robin patch sampling."""
    selected, pat_counts, sld_counts = [], {p: 0 for p in patients}, {}
    prog = True
    while len(selected) < n and prog:
        prog = False
        for p in patients:
            if len(selected) >= n or pat_counts[p] >= max_p:
                continue
            for s in h[p]:
                if len(selected) >= n or pat_counts[p] >= max_p:
                    break
                if sld_counts.get(s, 0) >= max_s or not h[p][s]:
                    continue
                selected.append(h[p][s].pop(0))
                pat_counts[p] += 1
                sld_counts[s] = sld_counts.get(s, 0) + 1
                prog = True
    return selected, pat_counts, sld_counts


def _contribution_cap(n_examples: int, fraction: float, unit: str) -> int:
    """Return an exact contribution cap, rejecting allocations below one unit."""
    cap = int(np.floor(n_examples * fraction))
    if cap < 1:
        raise ValueError(
            f"{n_examples} examples cannot satisfy the {fraction:.0%} {unit} cap"
        )
    return cap


def select_patches_round_robin(
    df_class: pd.DataFrame, n_patches: int, seed: int
) -> pd.DataFrame:
    """Sample patches with round-robin patient and slide caps (10% patient, 5% slide)."""
    if df_class.empty or n_patches <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    patients, h = _build_patch_hierarchy(df_class, rng)
    selected, _, _ = _loop_patches(
        patients,
        h,
        _contribution_cap(n_patches, 0.10, "patient"),
        _contribution_cap(n_patches, 0.05, "slide"),
        n_patches,
    )
    if len(selected) < n_patches:
        raise ValueError(
            "Patch allocation is infeasible under the 10% patient and 5% slide caps"
        )
    return df_class.loc[selected]


def _loop_slides(patients: list[str], h: dict, max_p: int, n: int) -> list[int]:
    """Execute loop for round robin slide sampling."""
    selected, pat_counts = [], {p: 0 for p in patients}
    prog = True
    while len(selected) < n and prog:
        prog = False
        for p in patients:
            if len(selected) >= n or pat_counts[p] >= max_p:
                continue
            if h[p]:
                selected.append(h[p].pop(0))
                pat_counts[p] += 1
                prog = True
    return selected


def _build_slide_hierarchy(
    df_slides: pd.DataFrame, rng: np.random.Generator
) -> tuple[list[str], dict[str, list[int]]]:
    """Build a randomized per-patient slide-index dictionary."""
    patients = list(cast(np.ndarray, df_slides["case_id"].unique()))
    rng.shuffle(patients)
    h = {
        p: list(df_slides[df_slides["case_id"] == p].index.to_numpy()) for p in patients
    }
    for p in h:
        rng.shuffle(h[p])
    return patients, h


def select_slides_round_robin(
    df_class: pd.DataFrame, n_slides: int, seed: int
) -> pd.DataFrame:
    """Sample slides for MIL with round-robin patient caps (10%)."""
    if df_class.empty or n_slides <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    df_slides = df_class.drop_duplicates("slide_id")
    patients, h = _build_slide_hierarchy(df_slides, rng)
    selected = _loop_slides(
        patients, h, _contribution_cap(n_slides, 0.10, "patient"), n_slides
    )
    if len(selected) < n_slides:
        raise ValueError("Slide allocation is infeasible under the 10% patient cap")
    return cast(
        pd.DataFrame,
        df_class[df_class["slide_id"].isin(df_slides.loc[selected, "slide_id"])],
    )


def build_manifest_hash(manifest_df: pd.DataFrame) -> str:
    """Create a hash of key manifest columns for immutability verification."""
    columns = cast(
        pd.DataFrame, manifest_df[["case_id", "slide_id", "cancer_type", "split"]]
    )
    records = columns.sort_values(by=["split", "cancer_type", "slide_id"]).to_dict(
        "records"
    )
    return compute_data_hash(records)
