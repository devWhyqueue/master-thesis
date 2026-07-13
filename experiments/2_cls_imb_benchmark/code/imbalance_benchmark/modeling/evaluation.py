from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from imbalance_benchmark.modeling.models import AttentionMil, DualExpertMil

__all__ = ["run_evaluation", "per_class_recall"]


def per_class_recall(
    preds: np.ndarray, targets: np.ndarray, n_classes: int
) -> list[float]:
    """Compute per-class recall from prediction and target arrays."""
    return [
        float((preds[targets == c] == c).sum()) / max(1, int((targets == c).sum()))
        for c in range(n_classes)
    ]


def _gather_and_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_mil: bool,
    n_classes: int,
) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
    """Gather logits and compute balanced accuracy and F1."""
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            if is_mil:
                bags, targets = batch
                logits = (
                    model.forward_ensemble([b.to(device) for b in bags])
                    if isinstance(model, DualExpertMil)
                    else cast(AttentionMil, model).forward_bags(
                        [b.to(device) for b in bags]
                    )[0]
                )
            else:
                logits, targets = model(batch["features"].to(device)), batch["target"]
            all_logits.append(logits.cpu())
            all_targets.append(targets)
    logits, targets = torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0).long()
    preds = logits.softmax(dim=-1).argmax(dim=-1)
    recalls = [
        float((preds[targets == c] == c).sum().item())
        / max(1, (targets == c).sum().item())
        for c in range(n_classes)
    ]
    f1s = []
    for c in range(n_classes):
        tp = float(((preds == c) & (targets == c)).sum().item())
        fp = float(((preds == c) & (targets != c)).sum().item())
        fn = float(((preds != c) & (targets == c)).sum().item())
        f1s.append(2 * tp / max(1.0, 2 * tp + fp + fn))
    return (
        float(np.mean(recalls)),
        float(np.mean(f1s)),
        float(F.cross_entropy(logits, targets).item()),
        logits,
        targets,
    )


def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_mil: bool,
    n_classes: int,
    logit_adj_tau: float | None = None,
    class_priors: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Evaluate model and return predictions, probs, and validation metrics."""
    acc, f1, nll, logits, targets = _gather_and_eval(
        model, loader, device, is_mil, n_classes
    )
    if logit_adj_tau is not None and class_priors is not None:
        logits = logits - logit_adj_tau * torch.log(class_priors.cpu() + 1e-8)
    probs = torch.softmax(logits, dim=-1)
    return {
        "balanced_accuracy": acc,
        "macro_f1": f1,
        "nll": nll,
        "logits": logits.numpy(),
        "probs": probs.numpy(),
        "preds": probs.argmax(dim=-1).numpy(),
        "targets": targets.numpy(),
    }
