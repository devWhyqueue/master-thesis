import torch
from torch import nn
from torch.nn import functional as F


class SoftF1LossMulti(nn.Module):
    """Macro-averaged multiclass soft F1 loss."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return one minus macro soft F1."""
        probs = torch.softmax(logits, dim=1)
        true_positive = (labels * probs).sum(dim=0)
        false_positive = ((1 - labels) * probs).sum(dim=0)
        false_negative = (labels * (1 - probs)).sum(dim=0)
        precision = true_positive / (true_positive + false_positive + 1e-16)
        recall = true_positive / (true_positive + false_negative + 1e-16)
        soft_f1 = 2 * precision * recall / (precision + recall + 1e-16)
        return 1 - soft_f1.mean()


class SoftMCCLossMulti(nn.Module):
    """Multiclass soft Matthews correlation coefficient loss."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return one minus multiclass soft MCC."""
        probs = torch.softmax(logits, dim=1)
        correct = torch.sum(probs * labels)
        sample_count = probs.size(0)
        label_totals = torch.sum(labels, dim=0)
        pred_totals = torch.sum(probs, dim=0)
        numerator = correct * sample_count - (label_totals * pred_totals).sum()
        denominator = (
            torch.sqrt(sample_count**2 - pred_totals.square().sum())
            * torch.sqrt(sample_count**2 - label_totals.square().sum())
            + 1e-8
        )
        return 1 - numerator / denominator


class CrossEntropyMetricLoss(nn.Module):
    """Cross-entropy plus a differentiable macro metric loss."""

    def __init__(
        self,
        n_classes: int,
        metric: str,
        metric_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if metric not in {"f1", "mcc"}:
            raise ValueError(f"Unsupported metric loss: {metric}")
        self.ce = nn.CrossEntropyLoss()
        self.metric = metric
        self.metric_loss_weight = metric_loss_weight
        self.soft_f1 = SoftF1LossMulti()
        self.soft_mcc = SoftMCCLossMulti()
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return CE plus the selected metric-derived loss."""
        one_hot = F.one_hot(targets, num_classes=self.n_classes).float()
        metric_loss = (
            self.soft_f1(logits, one_hot)
            if self.metric == "f1"
            else self.soft_mcc(logits, one_hot)
        )
        return self.ce(logits, targets) + self.metric_loss_weight * metric_loss
