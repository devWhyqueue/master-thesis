from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import torch

from imbalance_benchmark.common import (
    sign_file,
    split_paths,
    write_json,
)
from imbalance_benchmark.modeling.workflows.tuning_aggregate import (
    summarize_tuning_cost,
)
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    select_candidate_payload,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    condition_is_reusable,
    expected_observations,
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
    selections, payloads = {}, []
    for method in methods:
        if method == "post_hoc_logit_adjustment":
            specs = [ShardSpec(condition, method, 0, phase)]
        else:
            specs = resolve_round_specs(
                root, condition, phase, method, grids[method], reduce_round.index
            )
        candidates = [
            load_candidate(root, spec, reduce_round.fingerprint, expected_observations)
            for spec in specs
        ]
        payloads.extend(candidates)
        selections[method] = (
            candidates[0]["selection"]
            if method == "post_hoc_logit_adjustment"
            else select_candidate_payload(candidates)["config"]
        )
    return selections, payloads


def write_base_selection(
    root: Path, condition: str, selections: dict[str, Any]
) -> Path:
    """Persist the signed base-method selection consumed by dependent shards."""
    path = root / "tuning_shards" / f"base_selections_{condition}.json"
    write_json(path, selections)
    sign_file(path)
    return path


def write_base_selections(
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    methods: tuple[str, ...],
    roster: tuple[str, ...],
    conditions: tuple[str, ...],
) -> None:
    """Reduce and sign every incomplete base-method condition."""
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    for condition in conditions:
        if condition_is_reusable(base, condition, roster, assignments):
            continue
        selected, _ = reduce_phase(
            base["data"],
            condition,
            "base",
            methods,
            freeze["method_grids"],
            ReduceRound(fingerprint),
            expected_observations(condition, assignments, freeze),
        )
        write_base_selection(base["data"], condition, selected)


def write_serial_cost(
    paths: dict[str, Path],
    started: float,
    search_cost: dict[str, float | int],
    condition: str | None,
) -> None:
    """Preserve cost output for the legacy serial tuning command."""
    elapsed = time.perf_counter() - started
    name = (
        f"tuning_search_cost_{condition}.json"
        if condition
        else "tuning_search_cost.json"
    )
    write_json(
        paths["data"] / name,
        {
            "wall_clock_seconds": elapsed,
            "accelerator_hours": elapsed / 3600 if torch.cuda.is_available() else 0.0,
            "peak_accelerator_memory_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0,
            **search_cost,
        },
    )


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


def write_final_selections(
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    base_methods: tuple[str, ...],
    dependent_methods: tuple[str, ...],
    conditions: tuple[str, ...],
) -> None:
    """Write the unchanged signed selection interface and parallel search costs."""
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    for condition in conditions:
        if condition_is_reusable(
            base, condition, (*base_methods, *dependent_methods), assignments
        ):
            continue
        _reduce_condition(
            base,
            freeze,
            fingerprint,
            base_methods,
            dependent_methods,
            assignments,
            condition,
        )


def _reduce_condition(
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    base_methods: tuple[str, ...],
    dependent_methods: tuple[str, ...],
    assignments: tuple[str, ...],
    condition: str,
) -> None:
    reduce_round = ReduceRound(fingerprint)
    base_selected, base_payloads = reduce_phase(
        base["data"],
        condition,
        "base",
        base_methods,
        freeze["method_grids"],
        reduce_round,
        expected_observations(condition, assignments, freeze),
    )
    dependent, dependent_payloads = reduce_phase(
        base["data"],
        condition,
        "dependent",
        dependent_methods,
        freeze["method_grids"],
        reduce_round,
        expected_observations(condition, assignments, freeze),
    )
    selected = {**base_selected, **dependent}
    scoped = ("native",) if condition in {"natural", "balanced"} else assignments
    output = {assignment: {} for assignment in assignments}
    for assignment in scoped:
        output[assignment][condition] = selected
    _write_condition_outputs(
        base, condition, output, combined_cost([*base_payloads, *dependent_payloads])
    )


def _write_condition_outputs(
    base: dict[str, Path],
    condition: str,
    selections: dict[str, dict[str, Any]],
    cost: dict[str, Any],
) -> None:
    for index in range(3):
        paths = split_paths(base, index)
        selection_path = paths["data"] / f"tuning_selections_{condition}.json"
        write_json(selection_path, selections)
        sign_file(selection_path)
        write_json(paths["data"] / f"tuning_search_cost_{condition}.json", cost)
