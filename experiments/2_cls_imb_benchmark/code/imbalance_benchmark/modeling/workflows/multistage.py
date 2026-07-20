from __future__ import annotations

import logging
import math
from typing import Any, cast

import torch.nn as nn

from imbalance_benchmark.modeling.models import AttentionMil, MLP
from imbalance_benchmark.modeling.context import resolve_update_budget
from imbalance_benchmark.modeling.training import (
    fit_model,
    resolve_batch_size,
)

logger = logging.getLogger(__name__)


def _freeze_and_reinit_classifier(
    model: nn.Module, is_mil: bool
) -> tuple[nn.Module, ...]:
    """Freeze cRT's representation and reinitialize only its classifier head."""
    if is_mil:
        attention_model = cast(AttentionMil, model)
        for parameter in attention_model.instance_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in attention_model.attention.parameters():
            parameter.requires_grad_(False)
        for parameter in attention_model.projector.parameters():
            parameter.requires_grad_(False)
        classifier, frozen = (
            attention_model.classifier,
            (
                attention_model.instance_encoder,
                attention_model.attention,
                attention_model.projector,
            ),
        )
    else:
        mlp = cast(MLP, model)
        representation = mlp.net[:-1]
        for parameter in representation.parameters():
            parameter.requires_grad_(False)
        classifier, frozen = cast(nn.Linear, mlp.net[-1]), (representation,)
    nn.init.xavier_uniform_(classifier.weight)
    nn.init.zeros_(classifier.bias)
    return frozen


def _record_multistage_outcome(
    context: dict[str, Any], first: dict[str, Any], final: dict[str, Any]
) -> None:
    """Copy exact exposure and selected-checkpoint provenance to the outer context."""
    context["processed_examples"] = first.get("processed_examples", 0) + final.get(
        "processed_examples", 0
    )
    context["processed_instances"] = first.get("processed_instances", 0) + final.get(
        "processed_instances", 0
    )
    context["selected_checkpoint_step"] = final["selected_checkpoint_step"]


def _freeze_teacher(teacher: nn.Module) -> None:
    """Switch the trained RankMix teacher to deterministic inference-only mode."""
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


def _load_state(model: nn.Module, state: dict[str, Any], device: Any) -> None:
    """Load a checkpoint state onto the active training device."""
    model.load_state_dict({key: value.to(device) for key, value in state.items()})


def _rankmix_footprint(teacher: nn.Module, student: nn.Module) -> int:
    """Return the joint teacher-plus-student parameter footprint used for RankMix."""
    return sum(parameter.numel() for parameter in teacher.parameters()) + sum(
        parameter.numel() for parameter in student.parameters()
    )


def fit_crt(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Train cRT's seeded CE stage, then its balanced frozen-representation stage."""
    batch_size = resolve_batch_size(ctx["config"], ctx["is_mil"])
    budget = resolve_update_budget(ctx, batch_size)
    stage_one_context = {
        **ctx,
        "model": ctx["model"],
        "method": "ce",
        "param_config": ctx["stage_one_config"],
    }
    state, _ = fit_model(stage_one_context, max_steps=budget)
    model = ctx["model"]
    _load_state(model, state, ctx["device"])
    stage_two_context = {
        **ctx,
        "model": model,
        "method": "balanced_sampling",
        "param_config": {"lr": ctx["param_config"]["lr"], "parameter": 1.0},
        "frozen_eval_modules": _freeze_and_reinit_classifier(model, ctx["is_mil"]),
    }
    result = fit_model(stage_two_context, max_steps=math.ceil(0.2 * budget))
    _record_multistage_outcome(ctx, stage_one_context, stage_two_context)
    return result


def fit_rankmix(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Train a CE teacher, then a reinitialized RankMix-inspired student."""
    budget = resolve_update_budget(ctx, resolve_batch_size(ctx["config"], True))
    teacher = ctx["model_factory"]()
    ctx["training_footprint_parameters"] = _rankmix_footprint(teacher, ctx["model"])
    teacher_context = {
        **ctx,
        "model": teacher,
        "method": "ce",
        "param_config": {"lr": ctx["param_config"]["lr"]},
    }
    teacher_state, _ = fit_model(teacher_context, max_steps=budget)
    _load_state(teacher, teacher_state, ctx["device"])
    _freeze_teacher(teacher)
    student_context = {
        **ctx,
        "model": ctx["model_factory"](),
        "method": "rankmix",
        "teacher": teacher,
    }
    result = fit_model(student_context, max_steps=budget)
    _record_multistage_outcome(ctx, teacher_context, student_context)
    return result
