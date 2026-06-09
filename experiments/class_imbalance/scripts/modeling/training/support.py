from pathlib import Path
from typing import cast

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from scripts.common import ensure_dirs, load_config
from scripts.modeling.mil.metrics import extra_metrics
from scripts.modeling.training.support_tiers import (
    load_dataset_slide_counts,
    tier_support_for_classes,
)


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)


def _metric_payload(
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
    class_names: list[str],
    tier_support: np.ndarray | None = None,
) -> dict[str, object]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=cast(str, 0)
    )
    precision = cast(np.ndarray, precision)
    recall = cast(np.ndarray, recall)
    f1 = cast(np.ndarray, f1)
    support = cast(np.ndarray, support)
    present = cast(np.ndarray, support > 0)
    resolved_tier_support = tier_support
    if resolved_tier_support is None:
        resolved_tier_support = _default_dataset_tier_support(class_names)
    if resolved_tier_support is None:
        resolved_tier_support = support
    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision[present])),
        "macro_recall": float(np.mean(recall[present])),
        "macro_f1": float(np.mean(f1[present])),
        "class_names": class_names,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(map(int, y_true)),
        "preds": list(map(int, y_pred)),
        "probabilities": probabilities,
    }
    payload.update(
        extra_metrics(
            y_true,
            probabilities,
            precision,
            recall,
            f1,
            support,
            len(class_names),
            resolved_tier_support,
        )
    )
    return payload


def _default_dataset_tier_support(class_names: list[str]) -> np.ndarray | None:
    paths = ensure_dirs(load_config(None))
    return dataset_tier_support(class_names, paths["tables"] / "class_distribution.csv")


def dataset_tier_support(class_names: list[str], table_path: Path) -> np.ndarray | None:
    """Return dataset slide counts for tier assignment when available."""
    if not table_path.exists():
        return None
    return tier_support_for_classes(class_names, load_dataset_slide_counts(table_path))
