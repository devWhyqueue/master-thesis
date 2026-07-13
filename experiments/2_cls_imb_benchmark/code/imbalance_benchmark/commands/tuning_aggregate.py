from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from imbalance_benchmark.datasets.data import TrainDataset
from imbalance_benchmark.modeling.context import Regime, build_training_ctx, get_grid_configs
from imbalance_benchmark.modeling.special_methods import fit_crt, fit_method
from imbalance_benchmark.modeling.training import class_priors, run_evaluation

__all__ = ["TuningScope", "tune_across_splits"]


@dataclass
class TuningScope:
    """One patient split's controlled train manifest and natural validation loader."""

    regime: Regime
    val_loader: torch.utils.data.DataLoader
    train_ds: TrainDataset


def _selection_key(metrics: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Aggregate every split and seed by BA, macro-F1, then lower NLL."""
    return (
        sum(metric[0] for metric in metrics) / len(metrics),
        sum(metric[1] for metric in metrics) / len(metrics),
        -sum(metric[2] for metric in metrics) / len(metrics),
    )


def _evaluate(
    method: str,
    cfg: dict[str, Any],
    scope: TuningScope,
    seed: int,
    stage_one_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one candidate/seed/split and return its checkpoint plus validation metrics."""
    ctx = build_training_ctx(
        method, scope.train_ds, scope.regime, seed, cfg, scope.val_loader
    )
    if stage_one_config is not None:
        ctx["stage_one_config"] = stage_one_config
        state, _ = fit_crt(ctx)
    else:
        state, _ = fit_method(ctx)
    ctx["model"].load_state_dict(state)
    metrics = run_evaluation(
        ctx["model"],
        scope.val_loader,
        scope.regime.device,
        scope.regime.is_mil,
        scope.regime.n_classes,
    )
    return state, metrics


def _select_trainable(
    method: str,
    scopes: list[TuningScope],
    seeds: list[int],
    stage_one_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select a trainable method's configuration across all required observations."""
    configs = get_grid_configs(method, scopes[0].regime.n_classes)
    best_cfg, best_key = configs[0], None
    for cfg in configs:
        observations = []
        for scope in scopes:
            for seed in seeds:
                _, result = _evaluate(method, cfg, scope, seed, stage_one_config)
                observations.append(
                    (result["balanced_accuracy"], result["macro_f1"], result["nll"])
                )
        key = _selection_key(observations)
        if best_key is None or key > best_key:
            best_key, best_cfg = key, cfg
    return best_cfg


def _select_post_hoc(
    ce_config: dict[str, Any], scopes: list[TuningScope], seeds: list[int]
) -> dict[str, Any]:
    """Select one post-hoc strength from all selected CE checkpoints."""
    taus = [cfg["parameter"] for cfg in get_grid_configs("post_hoc_logit_adjustment")]
    observations: dict[float, list[tuple[float, float, float]]] = {
        tau: [] for tau in taus
    }
    for scope in scopes:
        priors = class_priors(
            scope.train_ds.get_int_targets(), scope.regime.n_classes, scope.regime.device
        )
        for seed in seeds:
            state, _ = _evaluate("ce", ce_config, scope, seed)
            ctx = build_training_ctx(
                "ce", scope.train_ds, scope.regime, seed, ce_config, scope.val_loader
            )
            ctx["model"].load_state_dict(state)
            for tau in taus:
                result = run_evaluation(
                    ctx["model"],
                    scope.val_loader,
                    scope.regime.device,
                    scope.regime.is_mil,
                    scope.regime.n_classes,
                    tau,
                    priors,
                )
                observations[tau].append(
                    (result["balanced_accuracy"], result["macro_f1"], result["nll"])
                )
    return {"parameter": max(taus, key=lambda tau: _selection_key(observations[tau]))}


def tune_across_splits(
    methods: tuple[str, ...], scopes: list[TuningScope], seeds: list[int]
) -> dict[str, Any]:
    """Tune the method roster jointly across all split, assignment, and seed observations."""
    selections: dict[str, Any] = {}
    for method in methods:
        if method == "crt":
            selections[method] = _select_trainable(
                method, scopes, seeds, selections["ce"]
            )
        elif method == "post_hoc_logit_adjustment":
            selections[method] = _select_post_hoc(selections["ce"], scopes, seeds)
        else:
            selections[method] = _select_trainable(method, scopes, seeds)
    return selections
