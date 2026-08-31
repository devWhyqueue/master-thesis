"""One-off dense validation-trace gate for plan items 3/4 (checkpoint schedule,
tuning truncation) in ``after-a-first-run-linear-wave.md``.

Reuses BRACS's frozen split-0 tuning context to fit every base-phase roster
method's full hyperparameter grid under two representative controlled
conditions, at the first tuning seed, recording *every* traced step's
validation metrics rather than just the sparse selection checkpoints. Offline
replay (``dense_trace_replay.py``) uses these traces to measure how often
truncating the training budget, or switching to a log-spaced checkpoint
schedule, would change which candidate gets selected.

Not part of the ``imbalance_benchmark`` package and not registered as a CLI
command: this gathers one-time evidence for a protocol decision, not a
recurring pipeline stage. OKO is excluded -- its own two-stage training loop
(``modeling.oko``) is not wired for dense tracing. ~180 candidates per
condition make this too slow sequentially, so candidates are packed
``--parallel-fits`` at a time across spawned processes (item 2's pattern):
each worker builds its own scope/feature bank and forces cuda placement.

Usage (from experiments/2_benchmark_patch/code, inside the project environment):
    python dense_trace_gate.py --config ../configs/bracs_patch.yaml \
        --conditions balanced severe --trace-interval 1 --parallel-fits 8
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any

from imbalance_benchmark.commands.tuning import (
    _frozen_shard_context,
    _tuning_seeds,
    load_shard_scope,
)
from imbalance_benchmark.datasets.features.cache import reset_feature_bank
from imbalance_benchmark.modeling.special_methods import fit_method
from imbalance_benchmark.modeling.training.context import build_training_ctx
from imbalance_benchmark.modeling.workflows.tuning.aggregation.aggregate import (
    _frozen_grid,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import phase_methods

logger = logging.getLogger(__name__)

EXCLUDED_METHODS = {"oko"}
_PARALLEL_FIT_ENV = {"IMB_FEATURE_BANK_DEVICE": "cuda"}
_CHILD_STAGGER_SECONDS = 3.0

WorkItem = tuple[str, int]  # (method, candidate_index)


def _work_items(freeze: dict[str, Any], condition: str) -> list[WorkItem]:
    methods = [
        method
        for method in phase_methods(False, "base", condition)
        if method not in EXCLUDED_METHODS
    ]
    return [
        (method, index)
        for method in methods
        for index in range(len(freeze["method_grids"].get(method, [])))
    ]


def _chunk(items: list[WorkItem], workers: int) -> list[list[WorkItem]]:
    chunks: list[list[WorkItem]] = [[] for _ in range(workers)]
    for position, item in enumerate(items):
        chunks[position % workers].append(item)
    return [chunk for chunk in chunks if chunk]


def _run_candidate(
    method: str,
    candidate_index: int,
    config: dict[str, Any],
    scope: Any,
    seed: int,
    trace_interval: int,
) -> dict[str, Any]:
    ctx = build_training_ctx(
        method,
        scope.train_ds,
        scope.regime,
        seed,
        config,
        scope.val_loader,
        scope.example_budget,
    )
    ctx["record_exposure"] = False
    ctx["dense_trace"] = []
    ctx["dense_trace_interval"] = trace_interval
    started = time.perf_counter()
    _, best_acc = fit_method(ctx)
    return {
        "method": method,
        "candidate_index": candidate_index,
        "config": config,
        "seed": seed,
        "selected_checkpoint_step": ctx.get("selected_checkpoint_step"),
        "final_acc": best_acc,
        "elapsed_seconds": time.perf_counter() - started,
        "trace": ctx["dense_trace"],
    }


def _worker(
    condition: str,
    args: argparse.Namespace,
    seed: int,
    trace_interval: int,
    out_dir: Path,
    items: list[WorkItem],
) -> None:
    os.environ.update(_PARALLEL_FIT_ENV)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    base, _, _, _, _ = _frozen_shard_context(args, load_scopes=False)
    scope = load_shard_scope(args, base, condition, "native", 0, 0, [])
    for method, candidate_index in items:
        path = out_dir / f"{condition}__{method}__{candidate_index}.json"
        if path.exists():
            continue  # resumable: a TIMEOUT can be resubmitted with the same command
        config = _frozen_grid(scope.regime, method)[candidate_index]
        record = _run_candidate(
            method, candidate_index, config, scope, seed, trace_interval
        )
        path.write_text(json.dumps(record))
        logger.info(
            "done: %s/%s/%d acc=%.4f step=%s (%.1fs)",
            condition,
            method,
            candidate_index,
            record["final_acc"],
            record["selected_checkpoint_step"],
            record["elapsed_seconds"],
        )
    reset_feature_bank()


def _run_condition(
    condition: str,
    args: argparse.Namespace,
    freeze: dict[str, Any],
    seed: int,
    trace_interval: int,
    out_dir: Path,
    parallel_fits: int,
) -> None:
    items = _work_items(freeze, condition)
    workers = min(max(1, parallel_fits), len(items))
    if workers <= 1:
        _worker(condition, args, seed, trace_interval, out_dir, items)
        return
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_worker, args=(condition, args, seed, trace_interval, out_dir, chunk)
        )
        for chunk in _chunk(items, workers)
    ]
    for index, process in enumerate(processes):
        process.start()
        if index < len(processes) - 1:
            time.sleep(_CHILD_STAGGER_SECONDS)
    for process in processes:
        process.join()
    failed = [str(index) for index, process in enumerate(processes) if process.exitcode]
    if failed:
        raise RuntimeError(f"Dense trace workers failed: {', '.join(failed)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--conditions", nargs="+", default=["balanced", "severe"])
    parser.add_argument("--trace-interval", type=int, default=1)
    parser.add_argument("--parallel-fits", type=int, default=8)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the dense-trace gate slice for every requested condition, per the module docstring."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args()

    base, _, freeze, _, _ = _frozen_shard_context(args, load_scopes=False)
    seed = _tuning_seeds(freeze)[0]
    out_dir = (
        Path(args.out) if args.out else base["data"] / "diagnostics" / "dense_trace"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        _run_condition(
            condition,
            args,
            freeze,
            seed,
            args.trace_interval,
            out_dir,
            args.parallel_fits,
        )


if __name__ == "__main__":
    main()
