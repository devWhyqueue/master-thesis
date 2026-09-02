"""Shared spawn-worker and atomic-per-key-cache machinery for analyze-combine.

Lifted out of ``secondary_intervals.interval_cache`` (the first user of this
pattern) so ``aggregation.aggregate``'s crossed permutation p-value cache can
reuse the same worker fan-out and atomic-write-then-rename resume semantics.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

__all__ = [
    "worker_count",
    "chunk_keys",
    "run_and_join",
    "spawn_workers",
    "atomic_write_bytes",
    "write_npz_cache",
    "read_npz_cache",
    "write_json_cache",
    "read_json_cache",
]

K = TypeVar("K")


def worker_count() -> int:
    """Worker process count from the SLURM CPU allocation, else all cores."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    return max(1, int(slurm_cpus)) if slurm_cpus else os.cpu_count() or 1


def chunk_keys(keys: list[K], workers: int) -> list[list[K]]:
    """Split ``keys`` round-robin into up to ``workers`` non-empty chunks."""
    chunks: list[list[K]] = [[] for _ in range(workers)]
    for index, key in enumerate(keys):
        chunks[index % workers].append(key)
    return [chunk for chunk in chunks if chunk]


def run_and_join(processes: list[Any]) -> None:
    """Start, join, and require every worker process to have exited cleanly."""
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [str(index) for index, process in enumerate(processes) if process.exitcode]
    if failed:
        raise RuntimeError(f"Cache workers failed: {', '.join(failed)}")


def spawn_workers(
    pending: list[K], worker_fn: Callable[..., None], fixed_args: tuple[Any, ...]
) -> None:
    """Fan ``pending`` keys out over ``min(worker_count(), len(pending))`` spawned
    processes, each running ``worker_fn(chunk, *fixed_args)``.

    Uses the ``spawn`` start method unconditionally (one fresh CUDA-free
    interpreter per worker, and the only method available on Windows), so
    ``worker_fn`` and every value in ``fixed_args`` must be picklable.
    """
    workers = min(worker_count(), len(pending))
    if workers == 0:
        return
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=worker_fn, args=(chunk, *fixed_args))
        for chunk in chunk_keys(pending, workers)
    ]
    run_and_join(processes)


def atomic_write_bytes(path: Path, write_fn: Callable[[Any], None]) -> None:
    """Write via a same-directory temp file, then atomically rename into place.

    ``write_fn`` receives an open binary file handle for the temp file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as handle:
        write_fn(handle)
    os.replace(temporary, path)


def write_npz_cache(path: Path, values: dict[str, np.ndarray]) -> None:
    """Atomically write one NPZ cache entry."""
    atomic_write_bytes(path, lambda handle: np.savez(handle, **values))  # type: ignore


def read_npz_cache(path: Path) -> dict[str, np.ndarray]:
    """Load one NPZ cache entry written by :func:`write_npz_cache`."""
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def write_json_cache(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one JSON cache entry."""
    atomic_write_bytes(
        path, lambda handle: handle.write(json.dumps(payload).encode("utf-8"))
    )


def read_json_cache(path: Path) -> dict[str, Any]:
    """Load one JSON cache entry written by :func:`write_json_cache`."""
    return json.loads(path.read_text(encoding="utf-8"))
