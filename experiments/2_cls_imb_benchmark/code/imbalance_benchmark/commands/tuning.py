from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from imbalance_benchmark.common import ensure_dirs, load_config, write_json
from imbalance_benchmark.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    TrainDataset,
    bag_collate,
)
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.modeling.context import (
    CONDITIONS,
    Regime,
    build_training_ctx,
    get_grid_configs,
    roster_for_regime,
)
from imbalance_benchmark.modeling.special_methods import (
    fit_crt,
    fit_method,
    select_post_hoc_tau,
)
from imbalance_benchmark.modeling.training import class_priors, run_evaluation

__all__ = ["cmd_tune"]


def _tune_method(
    method: str,
    train_ds: TrainDataset,
    val_loader: torch.utils.data.DataLoader,
    regime: Regime,
    seeds: list[int],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Sweep a method's candidate grid over the tuning seeds; select by BA -> F1 -> NLL."""
    configs = get_grid_configs(method, regime.n_classes)
    best_cfg, best_key = configs[0], None
    representative_state = None
    for cfg in configs:
        metrics = []
        for i, seed in enumerate(seeds):
            ctx = build_training_ctx(method, train_ds, regime, seed, cfg, val_loader)
            state, _ = fit_method(ctx)
            ctx["model"].load_state_dict(state)
            m = run_evaluation(
                ctx["model"], val_loader, regime.device, regime.is_mil, regime.n_classes
            )
            metrics.append((m["balanced_accuracy"], m["macro_f1"], m["nll"]))
            if method == "ce" and i == 0:
                representative_state = state
        mean_ba = sum(x[0] for x in metrics) / len(metrics)
        mean_f1 = sum(x[1] for x in metrics) / len(metrics)
        mean_nll = sum(x[2] for x in metrics) / len(metrics)
        key = (mean_ba, mean_f1, -mean_nll)
        if best_key is None or key > best_key:
            best_key, best_cfg = key, cfg
    return best_cfg, representative_state


def _tune_crt(
    stage_one_config: dict[str, Any],
    train_ds: TrainDataset,
    val_loader: torch.utils.data.DataLoader,
    regime: Regime,
    seeds: list[int],
) -> dict[str, Any]:
    """Select cRT's stage-two classifier learning rate; stage one inherits the CE config."""
    best_cfg, best_key = None, None
    for cfg in get_grid_configs("crt"):
        metrics = []
        for seed in seeds:
            ctx = build_training_ctx("crt", train_ds, regime, seed, cfg, val_loader)
            ctx["stage_one_config"] = stage_one_config
            state, _ = fit_crt(ctx)
            ctx["model"].load_state_dict(state)
            m = run_evaluation(
                ctx["model"], val_loader, regime.device, regime.is_mil, regime.n_classes
            )
            metrics.append((m["balanced_accuracy"], m["macro_f1"], m["nll"]))
        mean_ba = sum(x[0] for x in metrics) / len(metrics)
        mean_f1 = sum(x[1] for x in metrics) / len(metrics)
        mean_nll = sum(x[2] for x in metrics) / len(metrics)
        key = (mean_ba, mean_f1, -mean_nll)
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
    """Select tau for post-hoc logit adjustment on the selected CE model; no retraining."""
    if ce_state is None:
        return {"parameter": 1.0}
    ctx = build_training_ctx("ce", train_ds, regime, 0, {"lr": 1e-3}, val_loader)
    ctx["model"].load_state_dict(ce_state)
    priors = class_priors(train_ds.get_int_targets(), regime.n_classes, regime.device)
    taus = [c["parameter"] for c in get_grid_configs("post_hoc_logit_adjustment")]
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


def _tune_condition(
    condition: str,
    methods: tuple[str, ...],
    paths: dict[str, Path],
    val_loader: torch.utils.data.DataLoader,
    regime: Regime,
    seeds: list[int],
) -> dict[str, Any]:
    """Tune every roster method against one imbalance condition's training manifest."""
    dataset_cls = BagFeatureDataset if regime.is_mil else ImbalanceDataset
    train_ds = dataset_cls(
        paths["data"] / f"manifest_{condition}.csv", device=regime.device
    )
    selections: dict[str, Any] = {}
    ce_state = None
    for method in methods:
        if method == "crt":
            selections["crt"] = _tune_crt(
                selections["ce"], train_ds, val_loader, regime, seeds
            )
        elif method == "post_hoc_logit_adjustment":
            selections[method] = _tune_post_hoc(ce_state, train_ds, val_loader, regime)
        else:
            cfg, state = _tune_method(method, train_ds, val_loader, regime, seeds)
            selections[method] = cfg
            if method == "ce":
                ce_state = state
    return selections


def _tuning_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]:
    """Load config, the natural-validation loader, and the regime for the tuning sweep."""
    config = load_config(args.config)
    paths = ensure_dirs(config)
    freeze_path = paths["data"] / "manifest_freeze.json"
    if freeze_path.exists():
        verify_manifest_freeze(json.loads(freeze_path.read_text()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    dataset_cls = BagFeatureDataset if is_mil else ImbalanceDataset
    val_ds = dataset_cls(
        paths["data"] / "manifest.csv", split_name="validation", device=device
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, collate_fn=bag_collate if is_mil else None
    )
    regime = Regime(device, config, val_ds.get_n_classes(), is_mil)
    return paths, regime, val_loader


def cmd_tune(args: argparse.Namespace) -> None:
    """Run the validation-only hyperparameter search for every roster method and condition."""
    paths, regime, val_loader = _tuning_inputs(args)
    seeds = [
        derive_seed(args.seed, "tuning_initialization_0"),
        derive_seed(args.seed, "tuning_initialization_1"),
    ]
    methods = roster_for_regime(regime.is_mil)
    conditions = (args.condition,) if getattr(args, "condition", None) else CONDITIONS
    selections = {
        cond: _tune_condition(cond, methods, paths, val_loader, regime, seeds)
        for cond in conditions
    }
    if getattr(args, "condition", None):
        write_json(
            paths["data"] / f"tuning_selections_{args.condition}.json", selections
        )
    else:
        write_json(paths["data"] / "tuning_selections.json", selections)
