from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import torch

from imbalance_benchmark.datasets.data import TrainDataset
from imbalance_benchmark.modeling.context import (
    Regime,
    build_training_ctx,
    param_counts,
)
from imbalance_benchmark.modeling.special_methods import fit_crt, fit_method
from imbalance_benchmark.modeling.training import class_priors, run_evaluation

__all__ = [
    "TuningScope",
    "tune_across_splits",
    "summarize_tuning_cost",
    "combined_cost",
]


@dataclass
class TuningScope:
    """One patient split's controlled train manifest and natural validation loader."""

    regime: Regime
    val_loader: torch.utils.data.DataLoader
    train_ds: TrainDataset
    cost_records: list[dict[str, int]] = field(default_factory=list)
    update_budget: int | None = None
    assignment: str = "native"
    split_index: int = 0


def _frozen_grid(regime: Regime, method: str) -> list[dict[str, Any]]:
    """Return one method's signed pre-tuning grid, refusing live-source fallback."""
    grid = regime.method_grids.get(method)
    if not grid:
        raise RuntimeError(f"Frozen candidate grid missing for method: {method}")
    return grid


def summarize_tuning_cost(cost_records: list[dict[str, int]]) -> dict[str, float | int]:
    """Aggregate realized search work across every candidate, split, and seed fit."""
    processed = sum(record["processed_examples"] for record in cost_records)
    processed_instances = sum(
        record.get("processed_instances", 0) for record in cost_records
    )
    unique = sum(record["unique_training_examples"] for record in cost_records)
    return {
        "processed_examples": processed,
        "processed_instances": processed_instances,
        "effective_passes_through_unique_examples": processed / max(unique, 1),
        "maximum_total_parameters": max(
            (record["total_parameters"] for record in cost_records), default=0
        ),
        "maximum_trainable_parameters": max(
            (record["trainable_parameters"] for record in cost_records), default=0
        ),
        "maximum_training_footprint_parameters": max(
            (record["training_footprint_parameters"] for record in cost_records),
            default=0,
        ),
    }


def _selection_key(
    metrics: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
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
        method,
        scope.train_ds,
        scope.regime,
        seed,
        cfg,
        scope.val_loader,
        scope.update_budget,
    )
    if stage_one_config is not None:
        ctx["stage_one_config"] = stage_one_config
        state, _ = fit_crt(ctx)
    else:
        state, _ = fit_method(ctx)
    counts = param_counts(ctx["model"])
    scope.cost_records.append(
        {
            "processed_examples": int(ctx["processed_examples"]),
            "processed_instances": int(ctx.get("processed_instances", 0)),
            "unique_training_examples": len(ctx["train_dataset"]),
            "total_parameters": counts["total_parameters"],
            "trainable_parameters": counts["trainable_parameters"],
            "training_footprint_parameters": int(
                ctx.get("training_footprint_parameters", counts["total_parameters"])
            ),
        }
    )
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
    configs = _frozen_grid(scopes[0].regime, method)
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


def _tau_metrics_summary(metrics: list[tuple[float, float, float]]) -> dict[str, float]:
    """Average one tau's balanced accuracy, macro-F1, and NLL across observations."""
    return {
        "balanced_accuracy": sum(metric[0] for metric in metrics) / len(metrics),
        "macro_f1": sum(metric[1] for metric in metrics) / len(metrics),
        "nll": sum(metric[2] for metric in metrics) / len(metrics),
    }


def _select_post_hoc(
    ce_config: dict[str, Any], scopes: list[TuningScope], seeds: list[int]
) -> dict[str, Any]:
    """Select one post-hoc strength from all selected CE checkpoints.

    Every tau's averaged metrics are persisted alongside the selected one,
    not just the winner, so the selection is reproducible from the signed
    tuning selection alone without rerunning evaluation.
    """
    taus = [
        cfg["parameter"]
        for cfg in _frozen_grid(scopes[0].regime, "post_hoc_logit_adjustment")
    ]
    observations: dict[float, list[tuple[float, float, float]]] = {
        tau: [] for tau in taus
    }
    for scope in scopes:
        priors = class_priors(
            scope.train_ds.get_int_targets(),
            scope.regime.n_classes,
            scope.regime.device,
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
    best_tau = max(taus, key=lambda tau: _selection_key(observations[tau]))
    return {
        "parameter": best_tau,
        "taus": {str(tau): _tau_metrics_summary(observations[tau]) for tau in taus},
    }


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


def combined_cost(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge parallel search cost without treating wall time as accelerator time."""
    records = [record for payload in payloads for record in payload["cost_records"]]
    starts = [float(payload["started_at"]) for payload in payloads]
    completions = [float(payload["completed_at"]) for payload in payloads]
    return {
        "wall_clock_seconds": max(completions) - min(starts),
        "accelerator_hours": sum(
            float(payload["accelerator_seconds"]) for payload in payloads
        )
        / 3600,
        "peak_accelerator_memory_bytes": max(
            int(payload["peak_accelerator_memory_bytes"]) for payload in payloads
        ),
        "hardware": _unique_hardware(payloads),
        **summarize_tuning_cost(records),
    }


def _unique_hardware(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hardware = {
        json.dumps(payload["hardware"], sort_keys=True): payload["hardware"]
        for payload in payloads
    }
    return [hardware[key] for key in sorted(hardware)]
