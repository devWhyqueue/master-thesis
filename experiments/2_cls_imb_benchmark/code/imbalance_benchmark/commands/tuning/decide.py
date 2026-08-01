from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import os
from typing import Any

from imbalance_benchmark.commands.tuning import _frozen_shard_context, _is_excluded
from imbalance_benchmark.hydra.job_resources import build_job
from imbalance_benchmark.hydra.queue import check_queue_cap
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch
from imbalance_benchmark.hydra.dependent_jobs import (
    dependent_round_zero_jobs,
    final_reduce_job,
)
from imbalance_benchmark.hydra.workflow import _submit_script
from imbalance_benchmark.commands.tuning.round_windows import Window, this_round_windows
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    load_round_grids,
    load_round_state,
    merge_round_state,
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
    candidate_array_size,
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
        ReduceRound(fingerprint, index=round_index),
        _expected_observations(condition, assignments, freeze),
    )
    return selections


def _dependent_phase_started(root: Any, condition: str) -> bool:
    try:
        state = load_round_state(root, condition)
    except RuntimeError:
        return False
    return "crt" in state or "post_hoc_logit_adjustment" in state


def cmd_tune_decide(args: argparse.Namespace) -> None:
    """After one round's shards complete, resolve, shift, lock, or advance the phase."""
    base, scopes, freeze, fingerprint = _frozen_shard_context(args)
    if any(_is_excluded(paths) for paths, _, _ in scopes):
        return
    is_mil = scopes[0][1].is_mil
    methods = phase_methods(is_mil, args.phase)
    selections = _reduce_this_round(
        base, freeze, args.condition, args.phase, args.round, methods, fingerprint
    )
    n_classes = len(freeze["class_names"])
    windows = this_round_windows(
        base["data"], args.condition, args.phase, args.round, methods, n_classes
    )
    states = {
        method: decide_next_round(method, selections[method], *windows[method])
        for method in selections
    }
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
    """Shift unresolved methods to another round, or start the dependent phase."""
    unresolved = {
        method: state
        for method, state in states.items()
        if not (state.resolved or state.tuning_limited)
    }
    if unresolved:
        check_queue_cap()
        _submit_next_round(base, config, config_path, args, unresolved, windows, submit)
        return
    if args.phase != "base":
        check_queue_cap()
        submit(config, config_path, final_reduce_job(config, args.condition))
        return
    if _dependent_phase_started(base["data"], args.condition):
        return
    # states only holds methods still active *this* round - CE's own
    # readiness must come from the persisted, cross-round lock instead.
    ce_state = load_round_state(base["data"], args.condition).get("ce", {})
    if ce_state.get("resolved") or ce_state.get("tuning_limited"):
        check_queue_cap()
        _start_dependent_phase(base, config, config_path, args, is_mil, submit)


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
    shard = replace(
        build_job(
            config,
            f"tune-decide-shard-{args.condition}-{args.phase}-r{next_round}",
            f"tune-shard --phase {args.phase} --condition {args.condition}"
            f" --round {next_round}",
            True,
            (),
            "tune_controlled",
            "tune",
        ),
        array_size=candidate_array_size(tuple(new_windows)),
    )
    shard_id = submit(config, config_path, shard)
    decide = build_job(
        config,
        f"tune-decide-{args.condition}-{args.phase}-r{next_round}",
        f"tune-decide --phase {args.phase} --condition {args.condition}"
        f" --round {next_round}",
        False,
        (shard_id,),
        "tune_decide",
        "tune_reduce",
    )
    submit(config, config_path, decide)


def _start_dependent_phase(
    base: dict[str, Any],
    config: dict[str, Any],
    config_path: str,
    args: argparse.Namespace,
    is_mil: bool,
    submit: Callable[[dict[str, Any], str | None, SlurmJob], str],
) -> None:
    """Submit the dependent phase's frozen round-0 jobs, now that CE is final."""
    jobs = dependent_round_zero_jobs(config, is_mil)
    job_ids = tuple(submit(config, config_path, job) for job in jobs)
    decide = build_job(
        config,
        f"tune-decide-{args.condition}-dependent-r0",
        f"tune-decide --phase dependent --condition {args.condition} --round 0",
        False,
        job_ids,
        "tune_decide",
        "tune_reduce",
    )
    submit(config, config_path, decide)
