"""Manifest distribution statistics shared by natural and controlled conditions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd

from imbalance_benchmark.common import compute_data_hash

ClassKey = TypeVar("ClassKey", str, int)


def evidence_pool_hash(train_df: pd.DataFrame, classes: list[str], is_mil: bool) -> str:
    """Hash the fixed per-class patient/slide evidence pools shared by conditions."""
    columns = ["cancer_type", "case_id", "slide_id"]
    if not is_mil and "patch_id" in train_df:
        columns.append("patch_id")
    pool = pd.DataFrame(train_df.loc[train_df["cancer_type"].isin(classes), columns])
    pool = cast(pd.DataFrame, pool.sort_values(by=columns))
    return compute_data_hash(pool.to_dict("records"))


def natural_contribution_stats(
    train_df: pd.DataFrame, is_mil: bool
) -> dict[str, dict[str, float | int]]:
    """Report full-eligible-pool support and largest-unit contributions by class."""
    stats = {}
    for class_name, rows in train_df.groupby("cancer_type"):
        slide_rows = rows.drop_duplicates("slide_id") if is_mil else rows
        n_units = len(slide_rows) if is_mil else len(rows)
        stats[str(class_name)] = {
            "n_patients": int(rows["case_id"].nunique()),
            "n_slides": int(rows["slide_id"].nunique()),
            "n_patches": int(len(rows)),
            "max_patient_contribution": float(
                slide_rows["case_id"].value_counts().iloc[0] / max(1, n_units)
            ),
            "max_slide_contribution": float(
                slide_rows["slide_id"].value_counts().iloc[0] / max(1, n_units)
            ),
            "pool_fraction_retained": 1.0,
        }
    return stats


def normalized_entropy(counts: list[int]) -> float:
    """Compute the report's inverted normalized entropy, 1 - H/log(K)."""
    total = sum(counts)
    if len(counts) <= 1 or total <= 0:
        return 0.0
    probabilities = np.asarray(counts, dtype=float) / total
    probabilities = probabilities[probabilities > 0]
    return 1.0 - float(-(probabilities * np.log(probabilities)).sum()) / np.log(
        len(counts)
    )


def achieved_rho(counts: Mapping[ClassKey, int]) -> float:
    """Compute the realized head/tail imbalance ratio from per-class counts."""
    values = [count for count in counts.values() if count > 0]
    return max(values) / min(values) if values else 1.0


def _distribution_statistics(rows: pd.DataFrame, level: str) -> dict[str, Any]:
    frame = (
        rows if level == "patch" else rows.drop_duplicates(["slide_id", "cancer_type"])
    )
    counts = {
        str(name): int(count)
        for name, count in frame["cancer_type"].value_counts().items()
    }
    return {
        "counts": counts,
        "achieved_rho": achieved_rho(counts),
        "normalized_entropy": normalized_entropy(list(counts.values())),
    }


def support_statistics(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Report the required realized patch- and slide-level distributions."""
    return {
        "patch": _distribution_statistics(rows, "patch"),
        "slide": _distribution_statistics(rows, "slide"),
    }
