"""Parallel, checkpointed crossed permutation p-value cache for `_apply_gates`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from imbalance_benchmark.analysis.aggregation.parallel_cache import (
    read_json_cache,
    spawn_workers,
    write_json_cache,
)

__all__ = ["fill_crossed_p_values"]

CrossedPValue = Callable[
    [dict[str, Any], dict[str, Path], dict[str, Any], int], float | None
]

_SLIM_ENTRY_FIELDS = (
    "assignment",
    "severity",
    "method",
    "gate",
    "gate_passed",
    "descriptive_only",
    "dominant",
    "matched_methods",
    "unmatched_methods",
)


def _slim_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Strip an aggregated entry down to what a crossed-permutation worker needs.

    Excludes ``bootstrap_effect`` (10,001 floats) and the other per-replicate
    arrays -- workers only ever call ``crossed_p_value``, which never reads them.
    """
    return {key: entry[key] for key in _SLIM_ENTRY_FIELDS if key in entry}


def _cache_dir(base_paths: dict[str, Path], seed: int) -> Path:
    return base_paths["data"] / "crossed_p_value_cache" / f"seed{seed}"


def _cache_path(cache_dir: Path, entry: dict[str, Any]) -> Path:
    return cache_dir / (
        f"{entry['assignment']}__{entry['severity']}__"
        f"{entry['method']}__{entry['gate']}.json"
    )


def _group_by_assignment_severity(
    entries: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault((entry["assignment"], entry["severity"]), []).append(entry)
    return list(groups.values())


def _compute_caches(
    groups: list[list[dict[str, Any]]],
    base_paths: dict[str, Path],
    config: dict[str, Any],
    seed: int,
    cache_dir: Path,
    crossed_p_value: CrossedPValue,
) -> None:
    """Worker entry point: compute and cache each assigned entry's p-value.

    ``groups`` keeps every entry sharing an ``(assignment, severity)`` CE
    record together in the same worker, so ``load_seed_predictions``'s
    per-process memoization actually hits across the group instead of
    reloading the same CE block for each entry.
    """
    for entry in (entry for group in groups for entry in group):
        path = _cache_path(cache_dir, entry)
        if path.exists():
            continue
        p_value = crossed_p_value(entry, base_paths, config, seed)
        write_json_cache(path, {"p_value": p_value})


def fill_crossed_p_values(
    entries: list[dict[str, Any]],
    base_paths: dict[str, Path],
    config: dict[str, Any],
    seed: int,
    crossed_p_value: CrossedPValue,
) -> None:
    """Set ``entry["p_value"]`` for every entry, computing and caching only misses."""
    cache_dir = _cache_dir(base_paths, seed)
    pending = [entry for entry in entries if not _cache_path(cache_dir, entry).exists()]
    if pending:
        groups = _group_by_assignment_severity(
            [_slim_entry(entry) for entry in pending]
        )
        spawn_workers(
            groups,
            _compute_caches,
            (base_paths, config, seed, cache_dir, crossed_p_value),
        )
    for entry in entries:
        entry["p_value"] = read_json_cache(_cache_path(cache_dir, entry))["p_value"]
