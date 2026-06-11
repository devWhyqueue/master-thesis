"""Classification metric payloads."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from common_code.metrics.calibration import calibration_metrics


def resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)


def classification_payload(
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
    class_names: list[str],
    *,
    tier_support: np.ndarray | None = None,
    include_tier_metrics: bool = False,
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
    resolved_tier_support = tier_support if tier_support is not None else support
    payload: dict[str, object] = {
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
    payload.update(calibration_metrics(y_true, probabilities, len(class_names)))
    if include_tier_metrics:
        payload["support_tier_metrics"] = _tier_metrics(
            precision, recall, f1, support, resolved_tier_support
        )
    return payload


def _tier_metrics(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
    tier_support: np.ndarray,
) -> dict[str, object]:
    order = np.argsort(tier_support)
    tiers = {
        "tail": order[:8],
        "body": order[8:-8] if len(order) > 16 else order,
        "head": order[-8:],
    }
    return {
        name: {
            "precision": float(np.mean(precision[index])),
            "recall": float(np.mean(recall[index])),
            "f1": float(np.mean(f1[index])),
            "support": int(np.sum(support[index])),
        }
        for name, index in tiers.items()
        if len(index) > 0
    }
