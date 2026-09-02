from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from imbalance_benchmark.analysis.aggregation.parallel_cache import (
    read_npz_cache,
    spawn_workers,
    write_npz_cache,
)
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.metrics import assign_tiers
from imbalance_benchmark.analysis.query import load_seed_predictions
from imbalance_benchmark.analysis.reporting.secondary_intervals.calibration_intervals import (
    _complete_result_keys,
)
from imbalance_benchmark.common import split_paths

__all__ = ["distributions_by_key"]

logger = logging.getLogger(__name__)


def _locked_tiers(
    paths: dict[str, Path], assignment: str, condition: str, class_names: list[str]
) -> dict[str, str]:
    freeze_path = paths["data"] / "manifest_freeze.json"
    if not freeze_path.exists():
        return {}
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    allocated = (
        freeze.get("assignment_conditions", {})
        .get(assignment, {})
        .get(condition, {})
        .get("allocated_counts", {})
    )
    if not allocated:
        return {}
    order = freeze.get("tail_assignments", {}).get(assignment, class_names)
    return assign_tiers(class_names, allocated, order)


def _split_distributions(
    base_paths: dict[str, Path],
    contexts: list[BootstrapContext],
    is_mil: bool,
    ordinal: bool,
    assignment: str,
    condition: str,
    method: str,
) -> list[dict[str, np.ndarray]]:
    distributions = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        record = load_seed_predictions(
            paths,
            condition,
            method,
            assignment,
            fields=("preds", "probs", "temperature_scaled_probs"),
        )
        if record is None:
            raise RuntimeError(
                f"Missing secondary endpoints for {assignment}/{condition}/{method}"
            )
        class_names = list(record["class_names"])
        tiers = _locked_tiers(paths, assignment, condition, class_names)
        context = contexts[index]
        current = context.secondary_distributions(
            np.asarray(record["labels"]),
            np.asarray(record["preds"]),
            np.asarray(record["probs"]),
            class_names,
            tiers,
            is_mil=is_mil,
            ordinal=ordinal,
        )
        scaled = context.probability_secondary_distributions(
            np.asarray(record["labels"]),
            np.asarray(record["temperature_scaled_probs"]),
            class_names,
            tiers,
        )
        current.update(
            {f"temperature_scaled_{name}": values for name, values in scaled.items()}
        )
        distributions.append(current)
    return distributions


def _average_split_values(
    split_values: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        endpoint: np.mean(
            np.stack([values[endpoint] for values in split_values]), axis=0
        )
        for endpoint in split_values[0]
    }


def _cache_dir(base_paths: dict[str, Path], n_replicates: int, seed: int) -> Path:
    # Namespaced by (n_replicates, seed) so a differently-configured rerun cannot
    # silently reuse another run's cached distributions.
    return (
        base_paths["data"] / "secondary_interval_cache" / f"n{n_replicates}_seed{seed}"
    )


def _cache_path(cache_dir: Path, key: tuple[str, str, str]) -> Path:
    assignment, condition, method = key
    return cache_dir / f"{assignment}__{condition}__{method}.npz"


def _compute_key_caches(
    keys: list[tuple[str, str, str]],
    base_paths: dict[str, Path],
    is_mil: bool,
    ordinal: bool,
    n_replicates: int,
    seed: int,
    cache_dir: Path,
) -> None:
    """Worker entry point: compute and cache each assigned key's distributions.

    Runs in its own process, so a crash or TIMEOUT after this only loses the
    keys still pending -- already-written caches are picked up on resume.
    """
    # Contexts depend only on the split (paths/is_mil/n_replicates/seed), never on
    # assignment/condition/method -- build the 3 once per worker, not once per key.
    contexts = [
        BootstrapContext(split_paths(base_paths, index), is_mil, n_replicates, seed)
        for index in range(3)
    ]
    for assignment, condition, method in keys:
        path = _cache_path(cache_dir, (assignment, condition, method))
        if path.exists():
            continue
        logger.info("interval: %s/%s/%s", assignment, condition, method)
        split_values = _split_distributions(
            base_paths, contexts, is_mil, ordinal, assignment, condition, method
        )
        write_npz_cache(path, _average_split_values(split_values))


def distributions_by_key(
    base_paths: dict[str, Path],
    is_mil: bool,
    ordinal: bool,
    n_replicates: int,
    seed: int,
) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    """Return every complete result key's endpoint distributions, cached per key.

    Uncached keys are computed by up to ``worker_count()`` spawned processes
    (one fresh CUDA-free interpreter each) and cached to disk immediately, so
    a crash or wall-clock TIMEOUT resumes from whatever is already on disk
    instead of restarting at the first key.
    """
    keys = sorted(_complete_result_keys(base_paths))
    cache_dir = _cache_dir(base_paths, n_replicates, seed)
    pending = [key for key in keys if not _cache_path(cache_dir, key).exists()]
    if pending:
        spawn_workers(
            pending,
            _compute_key_caches,
            (base_paths, is_mil, ordinal, n_replicates, seed, cache_dir),
        )
    return {key: read_npz_cache(_cache_path(cache_dir, key)) for key in keys}
