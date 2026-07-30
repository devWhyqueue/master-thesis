"""Reduce tuning shards into signed selections.

``CE_ANCHORED_METHODS`` each degenerate exactly to plain CE at
``parameter=0`` (zero reweighting, zero focal gamma, zero auxiliary
weight), so ``reduce_phase`` aliases CE's own same-lr metrics as that
free candidate instead of training a redundant strength-0 model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    select_candidate_payload,
)
from imbalance_benchmark.modeling.workflows.tuning.search_windows import (
    CE_ANCHORED_METHODS,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    load_candidate,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_rounds import (
    resolve_round_specs,
)


@dataclass(frozen=True)
class ReduceRound:
    """One reduce call's verification fingerprint and adaptive-search round index."""

    fingerprint: list[str] = field(default_factory=list)
    index: int = 0


@dataclass(frozen=True)
class ReduceLocation:
    """Where one method's shards live: data root, imbalance condition, and phase."""

    root: Path
    condition: str
    phase: str


def _ce_anchor_candidates(
    active_grid: list[dict[str, Any]],
    real_candidates: list[dict[str, Any]],
    ce_by_lr: dict[float, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Alias CE's metrics as a free ``parameter=0`` candidate per active lr.

    Skipped where this round already trained a real ``parameter=0``
    candidate at that lr, or CE has no candidate there yet - the alias is
    a bonus comparison point, never a requirement.
    """
    trained_lrs = {cfg["lr"] for cfg in active_grid}
    already_zero = {
        candidate["config"]["lr"]
        for candidate in real_candidates
        if candidate["config"].get("parameter") == 0.0
    }
    return [
        {"config": {"parameter": 0.0, "lr": lr}, "metrics": ce_by_lr[lr]["metrics"]}
        for lr in trained_lrs
        if lr in ce_by_lr and lr not in already_zero
    ]


def _reduce_method(
    location: ReduceLocation,
    method: str,
    grids: dict[str, list[dict[str, Any]]],
    reduce_round: ReduceRound,
    expected_observations: int | None,
    ce_by_lr: dict[float, dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> Any:
    """Reduce one method's shards, updating ``ce_by_lr`` and ``payloads`` in place."""
    root, condition, phase = location.root, location.condition, location.phase
    if method == "post_hoc_logit_adjustment":
        candidate = load_candidate(
            root,
            ShardSpec(condition, method, 0, phase),
            reduce_round.fingerprint,
            expected_observations,
        )
        payloads.append(candidate)
        return candidate["selection"]
    specs = resolve_round_specs(
        root, condition, phase, method, grids[method], reduce_round.index
    )
    candidates = [
        load_candidate(root, spec, reduce_round.fingerprint, expected_observations)
        for spec in specs
    ]
    payloads.extend(candidates)
    selection_pool = candidates
    if method in CE_ANCHORED_METHODS and ce_by_lr:
        selection_pool = candidates + _ce_anchor_candidates(
            grids[method], candidates, ce_by_lr
        )
    if method == "ce":
        ce_by_lr.update(
            {candidate["config"]["lr"]: candidate for candidate in candidates}
        )
    return select_candidate_payload(selection_pool)["config"]


def reduce_phase(
    root: Path,
    condition: str,
    phase: str,
    methods: tuple[str, ...],
    grids: dict[str, list[dict[str, Any]]],
    reduce_round: ReduceRound,
    expected_observations: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reduce every required shard for one condition, phase, and search round."""
    location = ReduceLocation(root, condition, phase)
    selections: dict[str, Any] = {}
    payloads: list[dict[str, Any]] = []
    ce_by_lr: dict[float, dict[str, Any]] = {}
    for method in methods:
        selections[method] = _reduce_method(
            location,
            method,
            grids,
            reduce_round,
            expected_observations,
            ce_by_lr,
            payloads,
        )
    return selections, payloads
