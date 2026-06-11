import numpy as np
import torch
import torch.nn as nn

from tcga_ut_imbalanced.losses.focal import FocalLoss
from tcga_ut_imbalanced.losses.metric import CrossEntropyMetricLoss


class LossFactory:
    @staticmethod
    def build(
        loss_type: str,
        gamma: float | None = None,
        alpha: str | None = None,
        n_classes: int | None = None,
        class_counts: np.ndarray | None = None,
        metric_loss_weight: float = 1.0,
    ) -> nn.Module:
        """Build a configured loss module."""
        weights = _class_weights(alpha, n_classes, class_counts)
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(weight=weights)
        if loss_type == "focal_loss":
            if gamma is None or gamma < 0:
                raise ValueError(f"Invalid gamma: {gamma}")
            return FocalLoss(gamma=gamma, alpha=weights)
        if loss_type == "ce_soft_f1":
            return CrossEntropyMetricLoss(
                _validate_n_classes(n_classes), "f1", metric_loss_weight
            )
        if loss_type == "ce_soft_mcc":
            return CrossEntropyMetricLoss(
                _validate_n_classes(n_classes), "mcc", metric_loss_weight
            )
        raise ValueError(f"Unknown loss type: {loss_type}")


def _class_weights(
    alpha: str | None,
    n_classes: int | None,
    class_counts: np.ndarray | None,
) -> torch.Tensor | None:
    if alpha == "uniform":
        valid_n_classes = _validate_n_classes(n_classes)
        return _rescale(torch.ones(valid_n_classes, dtype=torch.double))
    if alpha == "inverse_class_frequency":
        valid_class_counts = _validate_class_counts(class_counts)
        return _rescale(torch.tensor(1.0 / valid_class_counts))
    if alpha is None:
        return None
    raise ValueError(f"Unknown alpha type: {alpha}")


def _rescale(weights: torch.Tensor) -> torch.Tensor:
    return weights * (len(weights) / weights.sum())


def _validate_n_classes(n_classes: int | None) -> int:
    if n_classes is None or n_classes < 0:
        raise ValueError(f"Invalid n_classes: {n_classes}")
    return n_classes


def _validate_class_counts(class_counts: np.ndarray | None) -> np.ndarray:
    if class_counts is None or (class_counts < 0).any():
        raise ValueError(f"Invalid class_counts: {class_counts}")
    return class_counts
