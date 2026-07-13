from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FocalLoss",
    "SoftF1LossMulti",
    "SoftMCCLossMulti",
    "ScholzCombinedLoss",
    "cfal_loss",
    "supervised_contrastive_loss",
]


class FocalLoss(nn.Module):
    """Focal Loss with custom alpha weights."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        """Initialize Focal Loss."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Focal Loss."""
        log_probs = F.log_softmax(logits, dim=-1)
        pt = torch.exp(log_probs).gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt).clamp(min=1e-6) ** self.gamma) * log_probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[targets]
        return loss.mean()


class SoftF1LossMulti(nn.Module):
    """Multiclass soft F1 loss (macro-averaged)."""

    def __init__(self, n_classes: int) -> None:
        """Initialize SoftF1LossMulti."""
        super().__init__()
        self.n_classes = n_classes

    def forward(
        self, logits: torch.Tensor, targets_one_hot: torch.Tensor
    ) -> torch.Tensor:
        """Calculate soft macro F1 loss."""
        probs = torch.sigmoid(logits)
        loss = torch.tensor(0.0, device=logits.device)
        for c in range(self.n_classes):
            p, t = probs[:, c], targets_one_hot[:, c]
            tp, fp, fn = (p * t).sum(), (p * (1 - t)).sum(), ((1 - p) * t).sum()
            loss += 1.0 - (2 * tp / (2 * tp + fp + fn + 1e-16))
        return loss / self.n_classes


class SoftMCCLossMulti(nn.Module):
    """Multiclass soft MCC loss."""

    def forward(
        self, logits: torch.Tensor, targets_one_hot: torch.Tensor
    ) -> torch.Tensor:
        """Calculate soft multiclass MCC loss."""
        probs = torch.softmax(logits, dim=1)
        correct = torch.sum(probs * targets_one_hot)
        n = probs.shape[0]
        t_totals, p_totals = torch.sum(targets_one_hot, dim=0), torch.sum(probs, dim=0)
        num = correct * n - torch.sum(t_totals * p_totals)
        p_spr = torch.clamp(n**2 - p_totals.square().sum(), min=1e-8)
        t_spr = torch.clamp(n**2 - t_totals.square().sum(), min=1e-8)
        return 1.0 - num / (torch.sqrt(p_spr) * torch.sqrt(t_spr) + 1e-8)


class ScholzCombinedLoss(nn.Module):
    """CE + Weighted Soft Metric Loss."""

    def __init__(self, n_classes: int, metric: str = "f1", weight: float = 1.0) -> None:
        """Initialize ScholzCombinedLoss."""
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.weight = weight
        self.metric = metric
        self.soft_f1 = SoftF1LossMulti(n_classes) if metric == "f1" else None
        self.soft_mcc = SoftMCCLossMulti() if metric == "mcc" else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate CE + Soft Metric Loss."""
        ce_loss = self.ce(logits, targets)
        oh = F.one_hot(targets, num_classes=logits.shape[1]).float()
        if self.metric == "f1":
            assert self.soft_f1 is not None
            metric_loss = self.soft_f1(logits, oh)
        else:
            assert self.soft_mcc is not None
            metric_loss = self.soft_mcc(logits, oh)
        return ce_loss + self.weight * metric_loss


def cfal_loss(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    class_counts: np.ndarray,
    margin: float = 1.0,
    gamma: float = 2.0,
    beta: float = 0.999,
) -> torch.Tensor:
    """Compute Center-Focused Affinity Loss (CFAL)."""
    counts = np.maximum(class_counts, 1.0)
    eff = (1.0 - beta**counts) / (1.0 - beta)
    inv_eff = torch.tensor(1.0 / eff, dtype=torch.float32, device=features.device)
    inv_eff = inv_eff * (len(eff) / inv_eff.sum())

    aff = model(features)
    true_aff = aff[torch.arange(len(targets), device=targets.device), targets]
    margins = torch.relu(margin + aff - true_aff.unsqueeze(1))
    margin_term = margins.masked_fill(
        F.one_hot(targets, num_classes=aff.shape[1]).bool(), 0.0
    ).sum(dim=1)
    loss_cfal = (
        inv_eff[targets] * (1.0 - true_aff).clamp(min=0.0).pow(gamma) * margin_term
    ).mean()

    proto = F.normalize(
        cast(torch.Tensor, getattr(model, "prototypes")), dim=-1, eps=1e-8
    )
    if proto.shape[0] < 2:
        return loss_cfal
    pw = (proto.unsqueeze(0) - proto.unsqueeze(1)).square().sum(dim=-1)
    reg = pw[torch.triu(torch.ones_like(pw), diagonal=1).bool()].var()
    return loss_cfal + reg


def supervised_contrastive_loss(
    embeddings: torch.Tensor, targets: torch.Tensor, temperature: float = 0.1
) -> tuple[torch.Tensor, int]:
    """Supervised contrastive loss for SC-MIL."""
    logits = torch.matmul(embeddings, embeddings.T) / temperature
    pos = (targets.unsqueeze(0) == targets.unsqueeze(1)) & ~torch.eye(
        len(targets), dtype=torch.bool, device=targets.device
    )
    n_pairs = int(pos.sum().item())
    if n_pairs == 0:
        return torch.tensor(0.0, device=embeddings.device), 0
    log_prob = logits - torch.logsumexp(
        logits.masked_fill(
            torch.eye(len(targets), dtype=torch.bool, device=targets.device), -1e9
        ),
        dim=1,
        keepdim=True,
    )
    return -(log_prob * pos.float()).sum() / pos.float().sum(), n_pairs
