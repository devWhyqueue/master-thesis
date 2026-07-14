from __future__ import annotations

import time
from typing import Any

import torch

from imbalance_benchmark.datasets.data import TrainDataset
from imbalance_benchmark.modeling.context import build_training_ctx
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
    "confirm_post_hoc",
    "confirm_crt",
    "confirm_method",
    "_test_prediction_hash",
]


def _timed_fit(fit_fn: Any, ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Run one training orchestration and measure its wall-clock time."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    state, _ = fit_fn(ctx)
    return state, time.perf_counter() - start


def confirm_ce(
    cond: str, cfg: dict[str, Any], train_ds: TrainDataset, run: RunContext
) -> list[tuple[dict[str, Any], int]]:
    """Fit CE for every confirmation seed; return its checkpoints for post-hoc reuse."""
    states = []
    for seed in run.seeds:
        ctx = build_training_ctx("ce", train_ds, run, seed, cfg, run.val_loader)
        state, elapsed = _timed_fit(fit_method, ctx)
        states.append((state, int(ctx.get("selected_checkpoint_step", 0))))
        _run_and_record(cond, "ce", len(states) - 1, ctx, state, run, elapsed)
    return states


def confirm_post_hoc(
    cond: str,
    cfg: dict[str, Any],
    ce_states: list[tuple[dict[str, Any], int]],
    train_ds: TrainDataset,
    run: RunContext,
) -> None:
    """Reuse each seed's locked CE checkpoint under a post-hoc target-prior shift."""
    priors = class_priors(train_ds.get_int_targets(), run.n_classes, run.device)
    tau = float(cfg.get("parameter", 1.0))
    for i, (seed, (state, checkpoint_step)) in enumerate(
        zip(run.seeds, ce_states, strict=True)
    ):
        ctx = build_training_ctx(
            "ce", train_ds, run, seed, {"parameter": tau}, run.val_loader
        )
        ctx["selected_checkpoint_step"] = checkpoint_step
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
    cond: str,
    method: str,
    cfg: dict[str, Any],
    train_ds: TrainDataset,
    run: RunContext,
) -> None:
    """Fit one ordinary (single-orchestration) roster method for every confirmation seed."""
    for i, seed in enumerate(run.seeds):
        ctx = build_training_ctx(method, train_ds, run, seed, cfg, run.val_loader)
        state, elapsed = _timed_fit(fit_method, ctx)
        _run_and_record(cond, method, i, ctx, state, run, elapsed)
