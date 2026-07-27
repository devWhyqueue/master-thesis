from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from imbalance_benchmark.datasets.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    load_training_dataset,
)


def feature_frame(
    manifest: Path,
    split: str | None,
    is_mil: bool,
    class_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load fixed embeddings and integer targets from one frozen manifest partition."""
    dataset = load_training_dataset(manifest, is_mil, split, class_names=class_names)
    if is_mil:
        bags = cast(BagFeatureDataset, dataset)
        features = [np.r_[bag.mean(0).cpu(), bag.std(0).cpu()] for bag, _ in bags]
    else:
        patches = cast(ImbalanceDataset, dataset)
        features = [
            patches[index]["features"].cpu().numpy() for index in range(len(patches))
        ]
    return np.asarray(features), dataset.get_int_targets()


def feature_identity(
    manifest: Path,
    split: str | None,
    is_mil: bool,
    class_names: list[str] | None,
) -> pd.DataFrame:
    """Return identities in the same one-row-per-observation order as features."""
    dataset = load_training_dataset(manifest, is_mil, split, class_names=class_names)
    return cast(pd.DataFrame, dataset.df[["case_id", "slide_id"]]).reset_index(
        drop=True
    )
