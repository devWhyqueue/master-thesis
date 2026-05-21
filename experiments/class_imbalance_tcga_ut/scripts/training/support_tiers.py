from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HEAD_CLASS_COUNT = 8
TAIL_CLASS_COUNT = 8


def tier_indices(support: np.ndarray) -> dict[str, np.ndarray]:
    """Map head/body/tail tiers to class indices ordered by support."""
    order = np.argsort(support)
    return {
        "tail": order[:TAIL_CLASS_COUNT],
        "body": order[TAIL_CLASS_COUNT : len(order) - TAIL_CLASS_COUNT],
        "head": order[-HEAD_CLASS_COUNT:],
    }


def load_dataset_slide_counts(table_path: Path) -> dict[str, int]:
    """Load full TCGA-UT slide counts keyed by cancer type."""
    frame = pd.read_csv(table_path)
    return dict(
        zip(frame["cancer_type"], frame["n_slides"].astype(int), strict=True)
    )


def tier_support_for_classes(
    class_names: list[str], slide_counts: dict[str, int]
) -> np.ndarray:
    """Return dataset slide counts aligned with ``class_names``."""
    return np.array([slide_counts[name] for name in class_names], dtype=np.int64)


def class_tier_labels(
    class_names: list[str], slide_counts: dict[str, int]
) -> dict[str, str]:
    """Assign each class to head, body, or tail from dataset slide support."""
    support = tier_support_for_classes(class_names, slide_counts)
    labels: dict[str, str] = {}
    for tier_name, indices in tier_indices(support).items():
        for index in indices:
            labels[class_names[int(index)]] = tier_name
    return labels


def load_class_tier_labels(
    class_names: list[str], table_path: Path
) -> dict[str, str] | None:
    """Load predefined tier labels when the dataset table is available."""
    if not table_path.exists():
        return None
    return class_tier_labels(class_names, load_dataset_slide_counts(table_path))
