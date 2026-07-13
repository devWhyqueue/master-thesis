from __future__ import annotations

import time
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from imbalance_benchmark.analysis.calibration import (
    apply_target_prior_correction,
    balanced_decision_logits,
    estimate_prior,
    softmax,
)
from imbalance_benchmark.analysis.metrics import classification_payload
from imbalance_benchmark.common import write_run_record
from imbalance_benchmark.datasets.data import TrainDataset
from imbalance_benchmark.modeling.context import (
    Regime,
    build_training_ctx,
    param_counts,
    updates_for,
)
from imbalance_benchmark.modeling.special_methods import fit_crt, fit_method
from imbalance_benchmark.modeling.training import (
    class_priors,
    resolve_batch_size,
    run_evaluation,
    update_budget,
)

__all__ = [
    "RunContext",
    "confirm_ce",
    "confirm_post_hoc",
    "confirm_crt",
    "confirm_method",
]


@dataclass
class RunContext(Regime):
    """Shared per-condition confirmation inputs: locked val/test loaders, paths, and seeds."""

    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    paths: dict[str, Path]
    seeds: list[int]
    class_names: list[str]
    assignment: str


def _timed_fit(fit_fn: Any, ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Run one training orchestration and measure its wall-clock time."""
    start = time.perf_counter()
    state, _ = fit_fn(ctx)
    return state, time.perf_counter() - start


def _cost_payload(
    method: str, budget: int, batch_size: int, elapsed: float, model: torch.nn.Module
) -> dict[str, Any]:
    """Assemble one confirmation run's update/example/timing/parameter cost record."""
    updates = updates_for(method, budget)
    return {
        "updates": updates,
        "processed_examples": updates * batch_size,
        "wall_clock_seconds": elapsed,
        "examples_per_update": batch_size,
        **param_counts(model),
    }


def _environment_payload() -> dict[str, Any]:
    """Capture the executable environment required to reproduce a confirmation run."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _split_payload(
    res: dict[str, Any],
    class_names: list[str],
    method: str,
    tau: float,
    train_priors: torch.Tensor,
    target_priors: np.ndarray,
) -> dict[str, Any]:
    """Assemble one evaluated split's real classwise metrics and prediction arrays."""
    raw_logits = res["logits"]
    train_prior_array = train_priors.detach().cpu().numpy()
    decision_logits = balanced_decision_logits(
        raw_logits, method, tau, train_prior_array
    )
    target_logits = apply_target_prior_correction(
        raw_logits, method, tau, train_prior_array, target_priors
    )
    decision_probs, target_probs = softmax(decision_logits), softmax(target_logits)
    payload = classification_payload(
        res["targets"].tolist(),
        decision_probs.argmax(axis=1).tolist(),
        target_probs.tolist(),
        class_names,
    )
    payload["labels"] = res["targets"].tolist()
    payload["preds"] = decision_probs.argmax(axis=1).tolist()
    payload["probabilities"] = target_probs.tolist()
    payload["logits"] = target_logits.tolist()
    payload["raw_logits"] = raw_logits.tolist()
    payload["raw_probabilities"] = res["probs"].tolist()
    payload["balanced_decision_logits"] = decision_logits.tolist()
    payload["target_prior_logits"] = target_logits.tolist()
    payload["target_prior_probabilities"] = target_probs.tolist()
    return payload


def _run_and_record(
    cond: str,
    method: str,
    seed_idx: int,
    ctx: dict[str, Any],
    state: dict[str, Any],
    run: RunContext,
    elapsed: float,
    logit_adj_tau: float = 1.0,
    class_priors_tensor: torch.Tensor | None = None,
) -> None:
    """Evaluate a locked confirmation checkpoint on validation and test, and write its run record."""
    model = ctx["model"]
    model.load_state_dict({k: v.to(run.device) for k, v in state.items()})
    if class_priors_tensor is None:
        class_priors_tensor = class_priors(
            ctx["train_labels"], run.n_classes, run.device
        )
    raw_results = {
        name: run_evaluation(model, loader, run.device, run.is_mil, run.n_classes)
        for name, loader in (("validation", run.val_loader), ("test", run.test_loader))
    }
    target_priors = estimate_prior(raw_results["validation"]["targets"], run.n_classes)
    splits = {
        name: _split_payload(
            result,
            run.class_names,
            method,
            logit_adj_tau,
            class_priors_tensor,
            target_priors,
        )
        for name, result in raw_results.items()
    }
    batch_size = resolve_batch_size(run.config, run.is_mil)
    budget = update_budget(len(ctx["train_dataset"]), batch_size)
    write_run_record(
        run.paths["results"]
        / f"assignment={run.assignment}"
        / cond
        / method
        / f"seed={seed_idx}",
        {
            "benchmark": "wsi" if run.is_mil else "patch",
            "condition": cond,
            "assignment": run.assignment,
            "method": method,
            "seed": ctx["seed"],
            "class_names": run.class_names,
            "tuning_params": ctx["param_config"],
            "train_priors": class_priors_tensor.detach().cpu().tolist(),
            "target_priors": target_priors.tolist(),
            "cost": _cost_payload(method, budget, batch_size, elapsed, model),
            "environment": _environment_payload(),
            "splits": splits,
        },
    )


def confirm_ce(
    cond: str, cfg: dict[str, Any], train_ds: TrainDataset, run: RunContext
) -> list[dict[str, Any]]:
    """Fit CE for every confirmation seed; return its checkpoints for post-hoc reuse."""
    states = []
    for seed in run.seeds:
        ctx = build_training_ctx("ce", train_ds, run, seed, cfg, run.val_loader)
        state, elapsed = _timed_fit(fit_method, ctx)
        states.append(state)
        _run_and_record(cond, "ce", len(states) - 1, ctx, state, run, elapsed)
    return states


def confirm_post_hoc(
    cond: str,
    cfg: dict[str, Any],
    ce_states: list[dict[str, Any]],
    train_ds: TrainDataset,
    run: RunContext,
) -> None:
    """Reuse each seed's locked CE checkpoint under a post-hoc target-prior shift."""
    priors = class_priors(train_ds.get_int_targets(), run.n_classes, run.device)
    tau = float(cfg.get("parameter", 1.0))
    for i, (seed, state) in enumerate(zip(run.seeds, ce_states, strict=True)):
        ctx = build_training_ctx(
            "ce", train_ds, run, seed, {"lr": 1e-3}, run.val_loader
        )
        _run_and_record(
            cond, "post_hoc_logit_adjustment", i, ctx, state, run, 0.0, tau, priors
        )


def confirm_crt(
    cond: str,
    cfg: dict[str, Any],
    stage_one_config: dict[str, Any],
    train_ds: TrainDataset,
    run: RunContext,
) -> None:
    """Fit cRT (stage one inherits CE; stage two retrains only the classifier) per seed."""
    for i, seed in enumerate(run.seeds):
        ctx = build_training_ctx("crt", train_ds, run, seed, cfg, run.val_loader)
        ctx["stage_one_config"] = stage_one_config
        state, elapsed = _timed_fit(fit_crt, ctx)
        _run_and_record(cond, "crt", i, ctx, state, run, elapsed)


def confirm_method(
    cond: str, method: str, cfg: dict[str, Any], train_ds: TrainDataset, run: RunContext
) -> None:
    """Fit one ordinary (single-orchestration) roster method for every confirmation seed."""
    for i, seed in enumerate(run.seeds):
        ctx = build_training_ctx(method, train_ds, run, seed, cfg, run.val_loader)
        state, elapsed = _timed_fit(fit_method, ctx)
        _run_and_record(cond, method, i, ctx, state, run, elapsed)
