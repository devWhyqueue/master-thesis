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

from imbalance_benchmark.modeling.context import get_grid_configs
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    resolve_terminal_specs,
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
    """One reduce call's verification fingerprint and adaptive-search round index.

    ``accepted`` (one hash set per split) additionally admits shards written
    before a freeze amendment, per ``validate_shard_payload``; unused (round
    has no meaning) when reused as a terminal-phase fingerprint carrier.
    """

    fingerprint: list[str] = field(default_factory=list)
    index: int = 0
    accepted: list[set[str]] | None = None


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


def combine_selection(
    method: str,
    active_grid: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    ce_by_lr: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    """Pick this method's winning config, aliasing CE's free zero-strength point."""
    selection_pool = candidates
    if method in CE_ANCHORED_METHODS and ce_by_lr:
        selection_pool = candidates + _ce_anchor_candidates(
            active_grid, candidates, ce_by_lr
        )
    if method == "ce":
        ce_by_lr.update(
            {candidate["config"]["lr"]: candidate for candidate in candidates}
        )
    return select_candidate_payload(selection_pool)["config"]


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
            reduce_round.accepted,
        )
        payloads.append(candidate)
        return candidate["selection"]
    specs = resolve_round_specs(
        root, condition, phase, method, grids[method], reduce_round.index
    )
    candidates = [
        load_candidate(
            root,
            spec,
            reduce_round.fingerprint,
            expected_observations,
            reduce_round.accepted,
        )
        for spec in specs
    ]
    payloads.extend(candidates)
    return combine_selection(method, grids[method], candidates, ce_by_lr)


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


def terminal_active_grids(
    state: dict[str, Any], methods: tuple[str, ...], n_classes: int
) -> dict[str, list[dict[str, Any]]]:
    """Expand every resolved or tuning-limited method's terminal active window.

    ``state`` is the signed cross-round tuning lock (``tuning_round_state``).
    Only the audited-unbounded controls (focal, ce_soft_f1, ce_soft_mcc)
    adaptively shift a strength window there; a fixed-grid method (oko,
    weighted_ce, balanced_sampling) always has ``strength_window: None`` in
    ``state`` even though it trains a real parameter per candidate, so
    ``get_grid_configs`` is used to fall back to that method's full frozen
    grid instead of silently dropping the parameter dimension.
    """
    return {
        method: get_grid_configs(
            method,
            n_classes,
            state[method]["lr_window"],
            state[method].get("strength_window"),
        )
        for method in methods
        if method in state
    }


def _reduce_terminal_method(
    location: ReduceLocation,
    method: str,
    terminal_grids: dict[str, list[dict[str, Any]]],
    reduce_round: ReduceRound,
    expected_observations: int | None,
    ce_by_lr: dict[float, dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> Any:
    """Reduce one method's terminal active window through the cross-round registry."""
    root, condition, phase = location.root, location.condition, location.phase
    fingerprint, accepted = reduce_round.fingerprint, reduce_round.accepted
    if method == "post_hoc_logit_adjustment":
        candidate = load_candidate(
            root,
            ShardSpec(condition, method, 0, phase),
            fingerprint,
            expected_observations,
            accepted,
        )
        payloads.append(candidate)
        return candidate["selection"]
    active_grid = terminal_grids[method]
    specs = resolve_terminal_specs(root, condition, phase, method, active_grid)
    candidates = [
        load_candidate(root, spec, fingerprint, expected_observations, accepted)
        for spec in specs
    ]
    for candidate, config in zip(candidates, active_grid, strict=True):
        if candidate["config"] != config:
            raise RuntimeError(
                f"Terminal shard config mismatch for {method}: registry expected "
                f"{config}, shard has {candidate['config']}"
            )
    payloads.extend(candidates)
    return combine_selection(method, active_grid, candidates, ce_by_lr)


def reduce_terminal_phase(
    root: Path,
    condition: str,
    phase: str,
    methods: tuple[str, ...],
    terminal_grids: dict[str, list[dict[str, Any]]],
    reduce_round: ReduceRound,
    expected_observations: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reduce every method's terminal active window into the signed final selection."""
    location = ReduceLocation(root, condition, phase)
    selections: dict[str, Any] = {}
    payloads: list[dict[str, Any]] = []
    ce_by_lr: dict[float, dict[str, Any]] = {}
    for method in methods:
        selections[method] = _reduce_terminal_method(
            location,
            method,
            terminal_grids,
            reduce_round,
            expected_observations,
            ce_by_lr,
            payloads,
        )
    return selections, payloads
