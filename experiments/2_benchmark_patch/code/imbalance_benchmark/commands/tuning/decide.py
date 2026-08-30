from __future__ import annotations

import argparse
from collections.abc import Callable
import os
from typing import Any

from imbalance_benchmark.commands.tuning import _frozen_shard_context, _is_excluded
from imbalance_benchmark.hydra.job_resources import build_job
from imbalance_benchmark.hydra.queue import check_queue_cap
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch
from imbalance_benchmark.hydra.dependent_jobs import final_reduce_job
from imbalance_benchmark.hydra.workflow import _submit_script
from imbalance_benchmark.commands.tuning.round_windows import Window, this_round_windows
from imbalance_benchmark.modeling.context import roster_for_condition
from imbalance_benchmark.modeling.workflows.tuning.aggregation.candidate_registry import (
    load_round_grids,
    load_round_state,
    merge_round_state,
    tuning_locked,
    write_round_grids,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    expected_observations as _expected_observations,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    write_base_selection,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import (
    ReduceRound,
    reduce_phase,
)
from imbalance_benchmark.modeling.workflows.tuning.search_windows import expand_grid
from imbalance_benchmark.modeling.workflows.tuning.tuning_rounds import (
    RoundState,
    decide_next_round,
    round_payload,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    phase_methods,
)

__all__ = ["cmd_tune_decide"]


def _real_submit(config: dict[str, Any], config_path: str | None, job: SlurmJob) -> str:
    return _submit_script(render_sbatch(job, config, config_path), False)


def _reduce_this_round(
    base: dict[str, Any],
    freeze: dict[str, Any],
    condition: str,
    phase: str,
    round_index: int,
    methods: tuple[str, ...],
    fingerprint: list[str],
    accepted: list[set[str]],
) -> dict[str, Any]:
    """Reduce this round's shards and return each method's winning configuration."""
    if round_index == 0:
        grids = {method: freeze["method_grids"][method] for method in methods}
    else:
        round_grids = load_round_grids(base["data"], condition, phase)
        grids = {
            method: expand_grid(**round_grids["windows"][method])
            for method in methods
            if method in round_grids["windows"]
        }
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    selections, _ = reduce_phase(
        base["data"],
        condition,
        phase,
        tuple(grids),
        grids,
        ReduceRound(fingerprint, index=round_index, accepted=accepted),
        _expected_observations(condition, assignments, freeze),
    )
    return selections


def _dependent_phase_started(root: Any, condition: str) -> bool:
    try:
        state = load_round_state(root, condition)
    except RuntimeError:
        return False
    return "crt" in state or "post_hoc_logit_adjustment" in state


def _resolve_round_states(
    base: dict[str, Any],
    freeze: dict[str, Any],
    fingerprint: list[str],
    accepted: list[set[str]],
    args: argparse.Namespace,
    is_mil: bool,
) -> tuple[dict[str, Any], dict[str, RoundState], dict[str, Window]]:
    """Reduce this round's shards and decide each method's next-round state."""
    methods = phase_methods(is_mil, args.phase, args.condition)
    selections = _reduce_this_round(
        base,
        freeze,
        args.condition,
        args.phase,
        args.round,
        methods,
        fingerprint,
        accepted,
    )
    windows = this_round_windows(
        base["data"],
        args.condition,
        args.phase,
        args.round,
        methods,
        len(freeze["class_names"]),
    )
    states = {
        method: decide_next_round(method, selections[method], *windows[method])
        for method in selections
    }
    return selections, states, windows


def cmd_tune_decide(args: argparse.Namespace) -> None:
    """After one round's shards complete, resolve, shift, lock, or advance the phase."""
    base, scopes, freeze, fingerprint, accepted = _frozen_shard_context(args)
    if any(_is_excluded(paths) for paths, _, _ in scopes):
        return
    is_mil = scopes[0][1].is_mil
    selections, states, windows = _resolve_round_states(
        base, freeze, fingerprint, accepted, args, is_mil
    )
    merge_round_state(base["data"], args.condition, round_payload(states))
    if args.phase == "base":
        write_base_selection(base["data"], args.condition, selections)
    config_path = os.path.abspath(args.config)
    _advance(base, freeze["runtime_config"], config_path, args, states, windows, is_mil)


def _advance(
    base: dict[str, Any],
    config: dict[str, Any],
    config_path: str,
    args: argparse.Namespace,
    states: dict[str, RoundState],
    windows: dict[str, Window],
    is_mil: bool,
    submit: Callable[[dict[str, Any], str | None, SlurmJob], str] = _real_submit,
) -> None:
    """Shift unresolved methods to another round, start the dependent phase,
    or -- once every method in *both* phases is locked -- submit the final
    reduce. Base and dependent run as separate, asynchronous round chains,
    so either phase's decide call can be the one that finishes last; each
    checks the full cross-phase lock rather than assuming its own phase is
    the whole story, and only the one that actually finishes last submits.
    """
    unresolved = {
        method: state
        for method, state in states.items()
        if not (state.resolved or state.tuning_limited)
    }
    if unresolved:
        check_queue_cap()
        _submit_next_round(base, config, config_path, args, unresolved, windows, submit)
        return
    dependent_methods = phase_methods(is_mil, "dependent", args.condition)
    if (
        args.phase == "base"
        and dependent_methods
        and not _dependent_phase_started(base["data"], args.condition)
    ):
        # states only holds methods still active *this* round - CE's own
        # readiness must come from the persisted, cross-round lock instead.
        ce_state = load_round_state(base["data"], args.condition).get("ce", {})
        if ce_state.get("resolved") or ce_state.get("tuning_limited"):
            check_queue_cap()
            _start_dependent_phase(base, config, config_path, args, is_mil, submit)
        return
    roster = roster_for_condition(is_mil, args.condition)
    if tuning_locked(base["data"], args.condition, roster):
        check_queue_cap()
        submit(config, config_path, final_reduce_job(config, args.condition))


def _submit_next_round(
    base: dict[str, Any],
    config: dict[str, Any],
    config_path: str,
    args: argparse.Namespace,
    unresolved: dict[str, RoundState],
    windows: dict[str, Window],
    submit: Callable[[dict[str, Any], str | None, SlurmJob], str],
) -> None:
    """Write the next round's active windows and submit its shard array + decide."""
    next_round = args.round + 1
    new_windows = {
        method: {
            "lr_window": state.next_lr_window or windows[method][0],
            "strength_window": state.next_strength_window or windows[method][1],
        }
        for method, state in unresolved.items()
    }
    write_round_grids(base["data"], args.condition, args.phase, next_round, new_windows)
    submit(
        config,
        config_path,
        build_job(
            config,
            f"tune-wave-{args.condition}-{args.phase}-r{next_round}",
            f"tune-wave --phase {args.phase} --condition {args.condition} --round {next_round}",
            False,
            (),
            "tune_decide",
            "tune_reduce",
        ),
    )


def _start_dependent_phase(
    base: dict[str, Any],
    config: dict[str, Any],
    config_path: str,
    args: argparse.Namespace,
    is_mil: bool,
    submit: Callable[[dict[str, Any], str | None, SlurmJob], str],
) -> None:
    """Submit the CE-inherited search's frozen round-0 jobs, now that this
    condition's CE config is locked."""
    del base, is_mil
    submit(
        config,
        config_path,
        build_job(
            config,
            "tune-wave-dependent-r0",
            "tune-wave --phase dependent --group controlled --round 0",
            False,
            (),
            "tune_decide",
            "tune_reduce",
        ),
    )
