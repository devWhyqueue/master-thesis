from typing import cast

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from scripts.common import ensure_dirs, load_config
from scripts.analysis.results import connect, init_schema, load_class_distribution
from scripts.modeling.training.support_tiers import tier_support_for_classes
from common_code.metrics.calibration import calibration_metrics
from common_code.metrics.payload import classification_payload, resolve_device
from common_code.metrics.payload import _tier_metrics


def _resolve_device(configured: str) -> torch.device:
    return resolve_device(configured)


def _metric_payload(
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
    class_names: list[str],
    tier_support: np.ndarray | None = None,
) -> dict[str, object]:
    resolved_tier_support = tier_support
    if resolved_tier_support is None:
        resolved_tier_support = _default_dataset_tier_support(class_names)
    if resolved_tier_support is None:
        labels = list(range(len(class_names)))
        _, _, _, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=cast(str, 0)
        )
        resolved_tier_support = cast(np.ndarray, support)
    payload = classification_payload(
        y_true,
        y_pred,
        probabilities,
        class_names,
        tier_support=resolved_tier_support,
        include_tier_metrics=True,
    )
    return payload


def _default_dataset_tier_support(class_names: list[str]) -> np.ndarray | None:
    paths = ensure_dirs(load_config(None))
    connection = connect(paths["db"])
    init_schema(connection)
    distribution = load_class_distribution(connection, paths)
    connection.close()
    if distribution.empty:
        return None
    slide_counts = dict(
        zip(
            distribution["cancer_type"].astype(str),
            distribution["n_slides"].astype(int),
            strict=True,
        )
    )
    return tier_support_for_classes(class_names, slide_counts)


__all__ = ["_metric_payload", "_resolve_device"]
