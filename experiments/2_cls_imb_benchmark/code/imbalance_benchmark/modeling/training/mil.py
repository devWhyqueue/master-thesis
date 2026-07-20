from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from imbalance_benchmark.modeling.losses import (
    rankmix_bag_loss,
    supervised_contrastive_loss,
)

__all__: list[str] = []


def _record_sc_mil_diagnostics(
    ctx: dict[str, Any], targets: torch.Tensor, positive_pairs: int, valid_anchors: int
) -> None:
    represented_classes = int(targets.unique().numel())
    diagnostics = ctx.setdefault("method_diagnostics", {})
    diagnostics.setdefault("sc_mil_batch_diagnostics", []).append(
        {
            "valid_anchors": valid_anchors,
            "ordered_positive_pairs": positive_pairs,
            "represented_classes": represented_classes,
        }
    )
    for name, value in (
        ("sc_mil_positive_pairs", positive_pairs),
        ("sc_mil_valid_anchors", valid_anchors),
        ("sc_mil_represented_classes", represented_classes),
        ("sc_mil_batches", 1),
    ):
        diagnostics[name] = diagnostics.get(name, 0) + value


def _rankmix_step(
    ctx: dict[str, Any], bags: list[torch.Tensor], targets: torch.Tensor
) -> torch.Tensor:
    exposure: list[int] = []
    loss = rankmix_bag_loss(
        ctx["model"], ctx["teacher"], bags, targets, ctx["param"], exposure
    )[0]
    ctx["processed_instances"] = ctx.get("processed_instances", 0) + exposure[0]
    return loss


def _sc_mil_step(
    ctx: dict[str, Any],
    bags: list[torch.Tensor],
    targets: torch.Tensor,
    step: int,
    max_steps: int,
) -> torch.Tensor:
    logits, embeddings, _ = ctx["model"].forward_bags(bags)
    contrastive, positive_pairs, valid_anchors = supervised_contrastive_loss(
        ctx["model"].project_bag_embeddings(embeddings),
        targets,
        temperature=ctx["param"],
    )
    _record_sc_mil_diagnostics(ctx, targets, positive_pairs, valid_anchors)
    progress = step / max_steps
    return (1.0 - progress) * contrastive + progress * F.cross_entropy(logits, targets)


def _fit_mil_step(
    batch_data: Any, ctx: dict[str, Any], step: int, max_steps: int
) -> torch.Tensor:
    """Execute one MIL update while recording exact bag and instance exposure."""
    device, method = ctx["device"], ctx["method"]
    bags = [bag.to(device) for bag in batch_data[0]]
    targets = batch_data[1].to(device)
    ctx["processed_examples"] = ctx.get("processed_examples", 0) + len(targets)
    if method == "rankmix":
        return _rankmix_step(ctx, bags, targets)
    ctx["processed_instances"] = ctx.get("processed_instances", 0) + sum(
        len(bag) for bag in bags
    )
    if method == "sc_mil":
        return _sc_mil_step(ctx, bags, targets, step, max_steps)
    logits, _, _ = ctx["model"].forward_bags(bags)
    if method == "logit_adjustment" and ctx["param"] is not None:
        logits = logits + ctx["param"] * torch.log(ctx["priors"] + 1e-8)
    return ctx["criterion"](logits, targets)
