from __future__ import annotations

import hashlib
from typing import cast

from pathlib import Path

import pandas as pd

from imbalance_benchmark.common import compute_data_hash, compute_sha256


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
