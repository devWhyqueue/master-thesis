from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from imbalance_benchmark.datasets import features as feature_lib
from imbalance_benchmark.datasets.feature_provenance import (
    VIRCHOW2_REVISION,
    VIRCHOW2_WEIGHTS_SHA256,
    resolve_feature_provenance,
    validate_feature_cache,
)
from imbalance_benchmark.datasets.features.cache_manifest import (
    cached_slide_ids,
    merge_pending_slides,
    record_pending_slide,
    save_tensor_atomic,
    validate_cached_slide,
)

__all__ = ["attach_extracted_features"]

SlideWork = tuple[str, list[str], list[str]]


def attach_extracted_features(
    frame: pd.DataFrame,
    feature_root: Path,
    feature_cfg: dict[str, Any] | None = None,
    device: torch.device | None = None,
    gpu_workers: int = 1,
) -> pd.DataFrame:
    """Extract ordered per-slide tensors, using configured visible GPUs when possible."""
    options = _extraction_options(feature_cfg or {}, device)
    _prepare_feature_cache(feature_root, options)
    enriched = frame.copy()
    paths, indices = _feature_references(enriched, feature_root, options, gpu_workers)
    enriched["feature_path"] = paths
    enriched["feature_index"] = indices.astype(int)
    return enriched


def _extraction_options(
    config: dict[str, Any], device: torch.device | None
) -> dict[str, Any]:
    return {
        "model_name": config.get("model_name", "hf-hub:paige-ai/Virchow2"),
        "batch_size": int(config.get("batch_size", 64)),
        "dtype": config.get("dtype", "float16"),
        "device": device,
        "revision": config.get("revision", VIRCHOW2_REVISION),
        "weights_sha256": config.get("weights_sha256", VIRCHOW2_WEIGHTS_SHA256),
    }


def _prepare_feature_cache(feature_root: Path, options: dict[str, Any]) -> None:
    feature_root.mkdir(parents=True, exist_ok=True)
    validate_feature_cache(feature_root, resolve_feature_provenance(options))


def _feature_references(
    frame: pd.DataFrame,
    feature_root: Path,
    options: dict[str, Any],
    requested_workers: int,
) -> tuple[pd.Series, pd.Series]:
    work = _slide_work(frame)
    expected = {
        slide_id: (feature_root / f"{slide_id}.pt", identities)
        for slide_id, _, identities in work
    }
    merge_pending_slides(feature_root, expected)
    _extract_missing_slides(work, feature_root, options, requested_workers)
    merge_pending_slides(feature_root, expected)
    return _references(frame, feature_root)


def _slide_work(frame: pd.DataFrame) -> list[SlideWork]:
    return [
        (
            str(slide_id),
            group["image_path"].astype(str).tolist(),
            _ordered_patch_identity(group),
        )
        for slide_id, group in frame.groupby("slide_id", sort=False)
    ]


def _extract_missing_slides(
    work: list[SlideWork],
    feature_root: Path,
    options: dict[str, Any],
    requested_workers: int,
) -> None:
    missing = _missing_work(work, feature_root)
    workers = _gpu_worker_count(requested_workers)
    if workers == 1:
        _extract_serial(missing, feature_root, options)
    else:
        _extract_parallel(missing, feature_root, options, workers)


def _missing_work(work: list[SlideWork], feature_root: Path) -> list[SlideWork]:
    cached = cached_slide_ids(feature_root)
    missing = []
    for item in work:
        slide_id, _, identities = item
        path = feature_root / f"{slide_id}.pt"
        if slide_id in cached:
            validate_cached_slide(
                feature_root, slide_id, path, identities, len(identities)
            )
        else:
            missing.append(item)
    return missing


def _gpu_worker_count(requested_workers: int) -> int:
    if requested_workers <= 1 or not torch.cuda.is_available():
        return 1
    return max(1, min(requested_workers, torch.cuda.device_count()))


def _extract_serial(
    work: list[SlideWork], feature_root: Path, options: dict[str, Any]
) -> None:
    model_cache: dict[str, Any] = {}
    for item in work:
        _extract_one(item, feature_root, options, model_cache)


def _extract_parallel(
    work: list[SlideWork],
    feature_root: Path,
    options: dict[str, Any],
    workers: int,
) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_extract_worker,
            args=(index, assignment, feature_root, options, workers),
        )
        for index, assignment in enumerate(_balanced_assignments(work, workers))
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [str(index) for index, process in enumerate(processes) if process.exitcode]
    if failed:
        raise RuntimeError(f"Feature extraction workers failed: {', '.join(failed)}")


def _extract_worker(
    worker_index: int,
    work: list[SlideWork],
    feature_root: Path,
    options: dict[str, Any],
    workers: int,
) -> None:
    worker_options = {
        **options,
        "device": torch.device("cuda", worker_index),
        "loader_workers": _loader_workers(workers),
    }
    model_cache: dict[str, Any] = {}
    for item in work:
        _extract_one(item, feature_root, worker_options, model_cache)


def _extract_one(
    work: SlideWork,
    feature_root: Path,
    options: dict[str, Any],
    model_cache: dict[str, Any],
) -> None:
    slide_id, image_paths, identities = work
    path = feature_root / f"{slide_id}.pt"
    tensor = feature_lib.extract_slide_features(image_paths, options, model_cache)
    save_tensor_atomic(tensor, path)
    record_pending_slide(feature_root, slide_id, path, identities, len(tensor))


def _balanced_assignments(work: list[SlideWork], workers: int) -> list[list[SlideWork]]:
    assignments = [[] for _ in range(workers)]
    loads = [0] * workers
    for item in sorted(work, key=lambda item: (-len(item[1]), item[0])):
        index = min(range(workers), key=lambda value: (loads[value], value))
        assignments[index].append(item)
        loads[index] += len(item[1])
    return assignments


def _loader_workers(worker_count: int) -> int:
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return max(0, min(allocated // worker_count - 1, 7))


def _references(frame: pd.DataFrame, feature_root: Path) -> tuple[pd.Series, pd.Series]:
    paths = pd.Series(index=frame.index, dtype=object)
    indices = pd.Series(index=frame.index, dtype=object)
    for slide_id, group in frame.groupby("slide_id", sort=False):
        paths.loc[group.index] = str(feature_root / f"{slide_id}.pt")
        indices.loc[group.index] = range(len(group))
    return paths, indices


def _ordered_patch_identity(group: pd.DataFrame) -> list[str]:
    patch_ids = (
        group["patch_id"].astype(str)
        if "patch_id" in group
        else group["image_path"].astype(str)
    )
    return [
        f"{patch_id}\0{image_path}"
        for patch_id, image_path in zip(
            patch_ids, group["image_path"].astype(str), strict=True
        )
    ]
