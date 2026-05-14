from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn

from scripts.mil.bags import AttentionMil, MdeMil


def bag_loss(
    method: str,
    model: nn.Module,
    bags: list[torch.Tensor],
    targets: torch.Tensor,
    train_labels: np.ndarray,
    weights: torch.Tensor,
    feature_gan_noise: bool = True,
) -> torch.Tensor:
    """Compute the selected feature-bag objective."""
    if method == "mde_mil":
        return _mde_loss(model, bags, targets, weights)
    attention_model = cast(AttentionMil, model)
    logits, embeddings, attention = attention_model.forward_bags(bags)
    if method == "rankmix_mil":
        logits, targets = _append_rankmix_examples(
            model, logits, embeddings, attention, targets
        )
    if method == "feature_gan_mil" and feature_gan_noise:
        logits, targets = _append_synthetic_tail_examples(
            model, logits, embeddings, targets, train_labels
        )
    loss = nn.functional.cross_entropy(logits, targets, weight=weights)
    if method == "cfal_mil":
        loss = loss + 0.1 * _cfal_affinity_loss(embeddings, targets, weights)
    if method == "sc_mil":
        loss = loss + 0.1 * _supervised_contrastive_loss(embeddings, targets)
    return loss


def _mde_loss(
    model: nn.Module,
    bags: list[torch.Tensor],
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    logits, _, original_logits, rebalanced_logits = cast(MdeMil, model).forward_bags(
        bags
    )
    consistency = torch.mean((original_logits - rebalanced_logits.detach()) ** 2)
    return (
        nn.functional.cross_entropy(logits, targets, weight=weights) + 0.1 * consistency
    )


def _append_rankmix_examples(
    model: nn.Module,
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    attention: list[torch.Tensor],
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(targets) < 2:
        return logits, targets
    scores = torch.tensor(
        [float(weights.max().detach().cpu()) for weights in attention],
        device=logits.device,
    )
    tail = targets.bincount(minlength=int(targets.max()) + 1)[targets] <= 1
    mixable = torch.where(tail | (scores < scores.median()))[0]
    if len(mixable) < 2:
        return logits, targets
    perm = mixable[torch.randperm(len(mixable), device=logits.device)]
    same_class = targets[mixable] == targets[perm]
    if not bool(same_class.any()):
        return logits, targets
    mixed = 0.5 * embeddings[mixable] + 0.5 * embeddings[perm]
    mixed_logits = cast(AttentionMil, model).classifier(mixed[same_class])
    mixed_targets = targets[mixable][same_class]
    return torch.cat([logits, mixed_logits]), torch.cat([targets, mixed_targets])


def _append_synthetic_tail_examples(
    model: nn.Module,
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    train_labels: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = np.bincount(train_labels)
    median_count = np.median(counts[counts > 0])
    mask = torch.tensor(
        [counts[int(label)] <= median_count for label in targets],
        device=targets.device,
    )
    if int(mask.sum().item()) == 0:
        return logits, targets
    synthetic = embeddings[mask] + 0.05 * torch.randn_like(embeddings[mask])
    synthetic_logits = cast(AttentionMil, model).classifier(synthetic)
    return torch.cat([logits, synthetic_logits]), torch.cat([targets, targets[mask]])


def _cfal_affinity_loss(
    embeddings: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=embeddings.device)
    for class_id in targets.unique():
        mask = targets == class_id
        if int(mask.sum().item()) < 2:
            continue
        class_embeddings = embeddings[mask]
        center = class_embeddings.mean(dim=0, keepdim=True)
        distances = torch.norm(class_embeddings - center, dim=1)
        hard = distances >= distances.mean()
        local = distances[hard].mean() if bool(hard.any()) else distances.mean()
        loss = loss + weights[class_id] * local
    return loss / max(1, len(targets.unique()))


def _supervised_contrastive_loss(
    embeddings: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    normalized = nn.functional.normalize(embeddings, dim=1)
    logits = torch.matmul(normalized, normalized.T) / 0.2
    same = targets.unsqueeze(0) == targets.unsqueeze(1)
    self_mask = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    positive = same & ~self_mask
    if not bool(positive.any()):
        return torch.tensor(0.0, device=embeddings.device)
    log_prob = logits - torch.logsumexp(
        logits.masked_fill(self_mask, -1e9), dim=1, keepdim=True
    )
    return -(log_prob * positive.float()).sum() / positive.float().sum()
