from __future__ import annotations

import time
from typing import Any

import torch

from imbalance_benchmark.datasets.data import TrainDataset
from imbalance_benchmark.modeling.training.context import build_training_ctx
from imbalance_benchmark.modeling.special_methods import fit_crt, fit_method
from imbalance_benchmark.modeling.training import class_priors
from imbalance_benchmark.modeling.workflows.confirmation_helpers import (
    RunContext,
    _run_and_record,
    _test_prediction_hash,
)

__all__ = [
    "RunContext",
    "confirm_ce",
    "confirm_ce_seed",
    "confirm_post_hoc",
    "confirm_post_hoc_seed",
    "confirm_crt",
    "confirm_crt_seed",
    "confirm_method",
    "confirm_method_seed",
    "_test_prediction_hash",
]


def _training_context(
    method: str,
    condition: str,
    train_ds: TrainDataset,
    run: RunContext,
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build a confirmation context with the signed condition-level exposure budget."""
    budget_kind = "natural" if condition == "natural" else "controlled"
    budget = run.exposure_budgets.get(budget_kind)
    args = (method, train_ds, run, seed, cfg, run.val_loader)
    return (
        build_training_ctx(*args)
        if budget is None
        else build_training_ctx(*args, budget)
    )


def _timed_fit(fit_fn: Any, ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Run one training orchestration and measure its wall-clock time."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    state, _ = fit_fn(ctx)
    return state, time.perf_counter() - start


def confirm_ce_seed(
    cond: str,
    cfg: dict[str, Any],
    train_ds: TrainDataset,
    run: RunContext,
    seed_idx: int,
) -> tuple[dict[str, Any], int]:
    """Fit CE for one confirmation seed; return its checkpoint for post-hoc reuse."""
    ctx = _training_context("ce", cond, train_ds, run, run.seeds[seed_idx], cfg)
    state, elapsed = _timed_fit(fit_method, ctx)
    checkpoint_step = int(ctx.get("selected_checkpoint_step", 0))
    _run_and_record(cond, "ce", seed_idx, ctx, state, run, elapsed)
    return state, checkpoint_step


def confirm_ce(
    cond: str, cfg: dict[str, Any], train_ds: TrainDataset, run: RunContext
) -> list[tuple[dict[str, Any], int]]:
    """Fit CE for every confirmation seed; return its checkpoints for post-hoc reuse."""
    return [
        confirm_ce_seed(cond, cfg, train_ds, run, seed_idx)
        for seed_idx in range(len(run.seeds))
    ]


def confirm_post_hoc_seed(
    cond: str,
    cfg: dict[str, Any],
    ce_state: dict[str, Any],
    checkpoint_step: int,
    train_ds: TrainDataset,
    run: RunContext,
    seed_idx: int,
) -> None:
    """Reuse one seed's locked CE checkpoint under a post-hoc target-prior shift."""
    priors = class_priors(train_ds.get_int_targets(), run.n_classes, run.device)
    tau = float(cfg.get("parameter", 1.0))
    ctx = _training_context(
        "ce", cond, train_ds, run, run.seeds[seed_idx], {"parameter": tau}
    )
    ctx["selected_checkpoint_step"] = checkpoint_step
    ctx["peak_memory_bytes"] = 0
    _run_and_record(
        cond,
        "post_hoc_logit_adjustment",
        seed_idx,
        ctx,
        ce_state,
        run,
        0.0,
        tau,
        priors,
    )


def confirm_post_hoc(
    cond: str,
    cfg: dict[str, Any],
    ce_states: list[tuple[dict[str, Any], int]],
    train_ds: TrainDataset,
    run: RunContext,
) -> None:
    """Reuse each seed's locked CE checkpoint under a post-hoc target-prior shift."""
    for seed_idx, (state, checkpoint_step) in enumerate(ce_states):
        confirm_post_hoc_seed(
            cond, cfg, state, checkpoint_step, train_ds, run, seed_idx
        )


def confirm_crt_seed(
    cond: str,
    cfg: dict[str, Any],
    stage_one_config: dict[str, Any],
    train_ds: TrainDataset,
    run: RunContext,
    seed_idx: int,
) -> None:
    """Fit cRT for one confirmation seed (stage one inherits CE; stage two retrains
    only the classifier)."""
    effective_config = {**cfg, "stage_one": stage_one_config}
    ctx = _training_context(
        "crt", cond, train_ds, run, run.seeds[seed_idx], effective_config
    )
    ctx["stage_one_config"] = stage_one_config
    state, elapsed = _timed_fit(fit_crt, ctx)
    _run_and_record(cond, "crt", seed_idx, ctx, state, run, elapsed)


def confirm_crt(
    cond: str,
    cfg: dict[str, Any],
    stage_one_config: dict[str, Any],
    train_ds: TrainDataset,
    run: RunContext,
) -> None:
    """Fit cRT (stage one inherits CE; stage two retrains only the classifier) per seed."""
    for seed_idx in range(len(run.seeds)):
        confirm_crt_seed(cond, cfg, stage_one_config, train_ds, run, seed_idx)


def confirm_method_seed(
    cond: str,
    method: str,
    cfg: dict[str, Any],
    train_ds: TrainDataset,
    run: RunContext,
    seed_idx: int,
) -> None:
    """Fit one ordinary (single-orchestration) roster method for one confirmation seed."""
    ctx = _training_context(method, cond, train_ds, run, run.seeds[seed_idx], cfg)
    state, elapsed = _timed_fit(fit_method, ctx)
    _run_and_record(
        cond,
        method,
        seed_idx,
        ctx,
        state,
        run,
        elapsed,
        float(cfg.get("parameter", 1.0)),
    )


def confirm_method(
    cond: str,
    method: str,
    cfg: dict[str, Any],
    train_ds: TrainDataset,
    run: RunContext,
) -> None:
    """Fit one ordinary (single-orchestration) roster method for every confirmation seed."""
    for seed_idx in range(len(run.seeds)):
        confirm_method_seed(cond, method, cfg, train_ds, run, seed_idx)
