from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar, cast
import numpy as np
import pandas as pd

from imbalance_benchmark.common import compute_sha256

__all__ = [
    "normalized_entropy",
    "achieved_rho",
    "contribution_stats",
    "build_tail_assignments",
    "verify_manifest_freeze",
]

ClassKey = TypeVar("ClassKey", str, int)


def normalized_entropy(counts: list[int]) -> float:
    """Compute the report's inverted normalized entropy, 1 - H/log(K) (Eq. 2)."""
    k = len(counts)
    total = sum(counts)
    if k <= 1 or total <= 0:
        return 0.0
    p = np.asarray(counts, dtype=float) / total
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum())
    return 1.0 - h / np.log(k)


def achieved_rho(counts: Mapping[ClassKey, int]) -> float:
    """Compute the realized head/tail imbalance ratio from per-class counts."""
    values = [c for c in counts.values() if c > 0]
    if not values:
        return 1.0
    return max(values) / min(values)


def _class_contribution_stats(
    rows: pd.DataFrame, pool_cls: pd.DataFrame, is_mil: bool
) -> dict[str, Any]:
    """Compute one class's unit counts, largest contributions, and pool coverage."""
    n_patients, n_slides = rows["case_id"].nunique(), rows["slide_id"].nunique()
    n_units = n_slides if is_mil else len(rows)
    slide_rows = rows.drop_duplicates("slide_id") if is_mil else rows
    patient_share = slide_rows["case_id"].value_counts().iloc[0] / max(1, n_units)
    slide_share = rows["slide_id"].value_counts().iloc[0] / max(1, len(rows))
    pool_denominator = (
        pool_cls["slide_id"].nunique() if is_mil else pool_cls["case_id"].nunique()
    )
    pool_numerator = n_slides if is_mil else n_patients
    return {
        "n_patients": int(n_patients),
        "n_slides": int(n_slides),
        "n_patches": int(len(rows)),
        "max_patient_contribution": float(patient_share),
        "max_slide_contribution": float(slide_share),
        "pool_fraction_retained": float(pool_numerator / max(1, pool_denominator)),
    }


def contribution_stats(
    condition_df: pd.DataFrame, eligible_pool: pd.DataFrame, is_mil: bool
) -> dict[str, dict[str, Any]]:
    """Report per-class unit counts, largest contributions, and pool coverage."""
    return {
        str(cls): _class_contribution_stats(
            rows,
            cast(pd.DataFrame, eligible_pool[eligible_pool["cancer_type"] == cls]),
            is_mil,
        )
        for cls, rows in condition_df.groupby("cancer_type")
    }


def build_tail_assignments(
    native_order: list[str], seed: int, ordinal: bool
) -> dict[str, list[str]]:
    """Build the three locked tail assignments: native, reversed/rotated, and random."""
    if len(native_order) == 2:
        return {"native": list(native_order), "reversed": list(reversed(native_order))}
    rotated_or_reversed = (
        list(reversed(native_order)) if ordinal else native_order[1:] + native_order[:1]
    )
    rng = np.random.default_rng(seed)
    random_order = list(rng.permutation(native_order))
    return {
        "native": list(native_order),
        "reversed_or_rotated": rotated_or_reversed,
        "random": random_order,
    }


def verify_manifest_freeze(freeze_meta: dict[str, Any]) -> None:
    """Refuse to proceed if any frozen condition manifest no longer matches its hash."""
    condition_sets = [freeze_meta.get("conditions", {})]
    condition_sets.extend(freeze_meta.get("assignment_conditions", {}).values())
    for conditions in condition_sets:
        for name, info in conditions.items():
            path = Path(info["path"])
            if not path.exists() or compute_sha256(path) != info["sha256"]:
                raise RuntimeError(
                    f"Manifest '{name}' at {path} no longer matches its frozen hash; "
                    "refusing to train on an altered condition."
                )
