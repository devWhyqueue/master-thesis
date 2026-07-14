from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from imbalance_benchmark.datasets.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    TrainDataset,
)
from imbalance_benchmark.modeling.context import (
    Regime,
    build_training_ctx,
    get_grid_configs,
)
from imbalance_benchmark.modeling.special_methods import (
    fit_crt,
    fit_method,
    select_post_hoc_tau,
)
from imbalance_benchmark.modeling.training import class_priors, run_evaluation


def tune_condition(
    methods: tuple[str, ...],
    val_loader: torch.utils.data.DataLoader,
    regime: Regime,
    seeds: list[int],
    manifest_path: Path,
) -> dict[str, Any]:
    """Tune every roster method against one imbalance condition's training manifest."""
    dataset_cls = BagFeatureDataset if regime.is_mil else ImbalanceDataset
    train_ds = dataset_cls(manifest_path, device=regime.device)
    selections: dict[str, Any] = {}
    ce_state = None
    for method in methods:
        selections[method], state = _tune_one(
            method, selections, ce_state, train_ds, val_loader, regime, seeds
        )
        if method == "ce":
            ce_state = state
    return selections


def _tune_one(
    method: str,
    selections: dict[str, Any],
    ce_state: dict[str, Any] | None,
    train_ds: TrainDataset,
    val_loader: torch.utils.data.DataLoader,
    regime: Regime,
    seeds: list[int],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Dispatch one method to its validation-only selection procedure."""
    if method == "crt":
        return _tune_crt(selections["ce"], train_ds, val_loader, regime, seeds), None
    if method == "post_hoc_logit_adjustment":
        return _tune_post_hoc(ce_state, train_ds, val_loader, regime), None
    return _tune_grid(method, train_ds, val_loader, regime, seeds)


def _tune_grid(
    method: str,
    train_ds: TrainDataset,
    val_loader: torch.utils.data.DataLoader,
    regime: Regime,
    seeds: list[int],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Select a grid configuration by validation BA, F1, then NLL."""
    best_cfg, best_key, representative_state = (
        get_grid_configs(method, regime.n_classes)[0],
        None,
        None,
    )
    for cfg in get_grid_configs(method, regime.n_classes):
        metrics = []
        for index, seed in enumerate(seeds):
            ctx = build_training_ctx(method, train_ds, regime, seed, cfg, val_loader)
            state, _ = fit_method(ctx)
            ctx["model"].load_state_dict(state)
            metric = run_evaluation(
                ctx["model"], val_loader, regime.device, regime.is_mil, regime.n_classes
            )
            metrics.append(
                (metric["balanced_accuracy"], metric["macro_f1"], metric["nll"])
            )
            if method == "ce" and index == 0:
                representative_state = state
        key = _selection_key(metrics)
        if best_key is None or key > best_key:
            best_key, best_cfg = key, cfg
    return best_cfg, representative_state


def _selection_key(
    metrics: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Compute the deterministic validation-selection ordering for a grid point."""
    mean_ba = sum(values[0] for values in metrics) / len(metrics)
    mean_f1 = sum(values[1] for values in metrics) / len(metrics)
    mean_nll = sum(values[2] for values in metrics) / len(metrics)
    return mean_ba, mean_f1, -mean_nll


def _tune_crt(
    stage_one_config: dict[str, Any],
    train_ds: TrainDataset,
    val_loader: torch.utils.data.DataLoader,
    regime: Regime,
    seeds: list[int],
) -> dict[str, Any]:
    """Select cRT's stage-two classifier learning rate."""
    best_cfg, best_key = None, None
    for cfg in get_grid_configs("crt"):
        metrics = []
        for seed in seeds:
            ctx = build_training_ctx("crt", train_ds, regime, seed, cfg, val_loader)
            ctx["stage_one_config"] = stage_one_config
            state, _ = fit_crt(ctx)
            ctx["model"].load_state_dict(state)
            metric = run_evaluation(
                ctx["model"], val_loader, regime.device, regime.is_mil, regime.n_classes
            )
            metrics.append(
                (metric["balanced_accuracy"], metric["macro_f1"], metric["nll"])
            )
        key = _selection_key(metrics)
        if best_key is None or key > best_key:
            best_key, best_cfg = key, cfg
    assert best_cfg is not None
    return best_cfg


def _tune_post_hoc(
    ce_state: dict[str, Any] | None,
    train_ds: TrainDataset,
    val_loader: torch.utils.data.DataLoader,
    regime: Regime,
) -> dict[str, Any]:
    """Select the post-hoc adjustment without retraining the CE model."""
    if ce_state is None:
        return {"parameter": 1.0}
    ctx = build_training_ctx("ce", train_ds, regime, 0, {"lr": 1e-3}, val_loader)
    ctx["model"].load_state_dict(ce_state)
    priors = class_priors(train_ds.get_int_targets(), regime.n_classes, regime.device)
    taus = [cfg["parameter"] for cfg in get_grid_configs("post_hoc_logit_adjustment")]
    best_tau, _ = select_post_hoc_tau(
        ctx["model"],
        val_loader,
        regime.device,
        regime.is_mil,
        regime.n_classes,
        priors,
        taus,
    )
    return {"parameter": best_tau}
