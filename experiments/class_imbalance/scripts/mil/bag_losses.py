from __future__ import annotations

from typing import cast

import torch
from torch import nn

from scripts.mil.bags import AttentionMil, DualExpertMil


def bag_loss(
    method: str,
    model: nn.Module,
    bags: list[torch.Tensor],
    targets: torch.Tensor,
    weights: torch.Tensor,
    progress: float,
    config: dict,
    teacher: AttentionMil | None = None,
    bags_b: list[torch.Tensor] | None = None,
    targets_b: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Compute a WSI-bag loss and return activation diagnostics."""
    if method == "mde_mil":
        if bags_b is None or targets_b is None:
            raise ValueError("MDE-MIL requires balanced and unbalanced batches.")
        return _mde_mil_loss(
            cast(DualExpertMil, model), bags, targets, bags_b, targets_b, config
        )
    attention_model = cast(AttentionMil, model)
    logits, embeddings, _ = attention_model.forward_bags(bags)
    if method == "rankmix_mil":
        return _rankmix_loss(attention_model, teacher, bags, targets, config)
    loss = _base_bag_loss(method, logits, targets, weights, config)
    if method == "sc_mil":
        return _sc_mil_loss(
            attention_model, embeddings, targets, loss, progress, config
        )
    return loss, {}


def _mde_mil_loss(
    model: DualExpertMil,
    bags_u: list[torch.Tensor],
    targets_u: torch.Tensor,
    bags_b: list[torch.Tensor],
    targets_b: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, int]]:
    embeddings_u = model.aggregate(bags_u)
    embeddings_b = model.aggregate(bags_b)
    logits_u = model.logits_u(embeddings_u)
    logits_b = model.logits_b(embeddings_b)
    logits_u_cross = model.logits_b(embeddings_u)
    logits_b_cross = model.logits_u(embeddings_b)
    loss_cls = nn.functional.cross_entropy(logits_u, targets_u) + nn.functional.cross_entropy(
        logits_b, targets_b
    )
    lambda_con = float(config["mde_mil_consistency_weight"])
    loss_con = nn.functional.mse_loss(logits_u, logits_u_cross) + nn.functional.mse_loss(
        logits_b, logits_b_cross
    )
    return loss_cls + lambda_con * loss_con, {
        "branch_u_batches": 1,
        "branch_b_batches": 1,
    }


def _base_bag_loss(
    method: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    if method == "mil_focal":
        return _focal_loss(logits, targets, float(config["focal_gamma"]))
    ce_weights = weights if method == "mil_weighted_ce" else None
    return nn.functional.cross_entropy(logits, targets, weight=ce_weights)


def _rankmix_loss(
    model: AttentionMil,
    teacher: AttentionMil | None,
    bags: list[torch.Tensor],
    targets: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, int]]:
    if teacher is None:
        raise ValueError("RankMix requires a frozen teacher model.")
    logits, soft_targets, diagnostics = _rankmix_batch(
        model, teacher, bags, targets, float(config["rankmix_alpha"])
    )
    return _soft_cross_entropy(logits, soft_targets), diagnostics


def _sc_mil_loss(
    model: AttentionMil,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    base_loss: torch.Tensor,
    progress: float,
    config: dict,
) -> tuple[torch.Tensor, dict[str, int]]:
    projections = model.project_bag_embeddings(embeddings)
    contrastive, positive_pairs = _supervised_contrastive_loss(
        projections, targets, float(config["sc_mil_temperature"])
    )
    beta = 1.0 - progress
    return beta * contrastive + (1.0 - beta) * base_loss, {
        "positive_pairs": positive_pairs
    }


def _focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    pt = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (-((1 - pt) ** gamma) * log_pt).mean()


def _rankmix_batch(
    model: AttentionMil,
    teacher: AttentionMil,
    bags: list[torch.Tensor],
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if len(bags) < 2:
        logits, _, _ = model.forward_bags(bags)
        return (
            logits,
            nn.functional.one_hot(targets, logits.shape[1]).float(),
            {"mixed_examples": 0},
        )
    permutation = torch.randperm(len(bags), device=targets.device)
    lambdas = (
        torch.distributions.Beta(alpha, alpha).sample((len(bags),)).to(targets.device)
    )
    mixed_bags: list[torch.Tensor] = []
    for idx, other_idx in enumerate(permutation.tolist()):
        mixed_bags.append(
            _mix_ranked_bags(
                teacher,
                bags[idx],
                bags[other_idx],
                int(targets[idx].item()),
                int(targets[other_idx].item()),
                float(lambdas[idx].item()),
            )
        )
    logits, _, _ = model.forward_bags(mixed_bags)
    one_hot = nn.functional.one_hot(targets, logits.shape[1]).float()
    mixed_targets = lambdas.unsqueeze(1) * one_hot + (1.0 - lambdas).unsqueeze(
        1
    ) * one_hot.index_select(0, permutation)
    return logits, mixed_targets, {"mixed_examples": len(mixed_bags)}


def _mix_ranked_bags(
    teacher: AttentionMil,
    first_bag: torch.Tensor,
    second_bag: torch.Tensor,
    first_class: int,
    second_class: int,
    mix_lambda: float,
) -> torch.Tensor:
    keep = min(len(first_bag), len(second_bag))
    first = _rank_representative_features(teacher, first_bag, first_class, keep)
    second = _rank_representative_features(teacher, second_bag, second_class, keep)
    return mix_lambda * first + (1.0 - mix_lambda) * second


def _rank_representative_features(
    teacher: AttentionMil, bag: torch.Tensor, class_id: int, keep: int
) -> torch.Tensor:
    scores = teacher.rank_scores(bag, class_id)
    ranked = torch.topk(scores, keep).indices
    original_order = torch.sort(ranked).values
    return bag.index_select(0, original_order)


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _supervised_contrastive_loss(
    embeddings: torch.Tensor, targets: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, int]:
    logits = torch.matmul(embeddings, embeddings.T) / temperature
    same = targets.unsqueeze(0) == targets.unsqueeze(1)
    self_mask = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    positive = same & ~self_mask
    positive_pairs = int(positive.sum().item())
    if positive_pairs == 0:
        return torch.tensor(0.0, device=embeddings.device), 0
    log_prob = logits - torch.logsumexp(
        logits.masked_fill(self_mask, -1e9), dim=1, keepdim=True
    )
    return -(log_prob * positive.float()).sum() / positive.float().sum(), positive_pairs
