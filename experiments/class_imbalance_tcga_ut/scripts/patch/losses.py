"""Patch classification losses, including Scholz et al. (MIDL 2024) objectives.

Scholz formulas match https://github.com/daniel-scholz/address-class-imbalance (CC-BY 4.0).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PatchFocalLoss(nn.Module):
    """Multiclass focal loss for patch classifiers."""

    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa
        """Compute the focal loss value."""
        log_probs = torch.log_softmax(logits, dim=1)
        pt = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (-((1 - pt) ** self.gamma) * log_pt).mean()


def inverse_frequency_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Build normalized inverse-frequency class weights."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    return torch.tensor(weights * (n_classes / weights.sum()), dtype=torch.float32)


class _SoftF1LossWithLogits(nn.Module):
    """Binary soft F1 loss on per-class probabilities."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return one minus soft F1 for a single binary problem."""
        pred = torch.sigmoid(pred)
        tp = (target * pred).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        precision = tp / (tp + fp + 1e-16)
        recall = tp / (tp + fn + 1e-16)
        soft_f1 = 2 * precision * recall / (precision + recall + 1e-16)
        return 1 - soft_f1


class SoftF1LossMulti(nn.Module):
    """Macro-averaged multiclass soft F1 loss."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._binary = _SoftF1LossWithLogits()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return one minus macro soft F1."""
        probs = torch.softmax(logits, dim=1)
        loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)
        for class_idx in range(self.num_classes):
            loss = loss + self._binary(logits[:, class_idx], labels[:, class_idx])
        return loss / self.num_classes


class SoftMCCLossMulti(nn.Module):
    """Multiclass soft MCC loss (with logits)."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return one minus multiclass soft MCC."""
        preds = torch.softmax(logits, dim=1)
        correct = torch.sum(preds * labels)
        sample_count = preds.size(0)
        label_totals = torch.sum(labels, dim=0)
        pred_totals = torch.sum(preds, dim=0)
        numerator = correct * sample_count - (label_totals * pred_totals).sum()
        denominator = (
            torch.sqrt(sample_count**2 - pred_totals.square().sum())
            * torch.sqrt(sample_count**2 - label_totals.square().sum())
            + 1e-8
        )
        return 1 - numerator / denominator


class ScholzCombinedLoss(nn.Module):
    """Equal-weight CE + soft F1 or soft MCC (Scholz et al., Sec. 3.1.4)."""

    def __init__(self, num_classes: int, metric: str) -> None:
        super().__init__()
        if metric not in {"f1", "mcc"}:
            raise ValueError(f"Unsupported Scholz metric: {metric}")
        self.ce = nn.CrossEntropyLoss()
        self.metric = metric
        self._soft_f1 = SoftF1LossMulti(num_classes) if metric == "f1" else None
        self._soft_mcc = SoftMCCLossMulti() if metric == "mcc" else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return CE plus the selected Scholz metric-derived loss."""
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        ce_loss = self.ce(logits, targets)
        if self.metric == "f1":
            assert self._soft_f1 is not None
            metric_loss = self._soft_f1(logits, one_hot)
        else:
            assert self._soft_mcc is not None
            metric_loss = self._soft_mcc(logits, one_hot)
        return ce_loss + metric_loss
