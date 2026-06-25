"""Validation-tuning grids for native external datasets."""

from __future__ import annotations

from dataclasses import dataclass

from common_code.tuning.grid import (
    PATCH_FEATURE_SPECS,
    SEEDS,
    TuningVariant,
    WSI_BAG_SPECS,
    grid_from_specs,
    validate_tuning_params as _validate_common,
)

PATCH_SPECS = tuple(
    spec for spec in PATCH_FEATURE_SPECS if spec[0] != "patch_feature_divide_conquer"
)
WSI_SPECS = WSI_BAG_SPECS


@dataclass(frozen=True)
class NativeTask:
    """One native-dataset tuning task."""

    variant: TuningVariant
    seed: int


def patch_grid() -> list[TuningVariant]:
    """Return the native patch-feature tuning grid."""
    return grid_from_specs("patch_feature", PATCH_SPECS)


def wsi_grid() -> list[TuningVariant]:
    """Return the native WSI-bag tuning grid."""
    return grid_from_specs("wsi_bag", WSI_SPECS)


def task_count(benchmark: str) -> int:
    """Return the total native tuning task count."""
    grid = patch_grid() if benchmark == "patch" else wsi_grid()
    return len(grid) * len(SEEDS)


def task_for_index(benchmark: str, index: int) -> NativeTask:
    """Return the native tuning task at a flat array index."""
    grid = patch_grid() if benchmark == "patch" else wsi_grid()
    tasks = [NativeTask(variant, seed) for variant in grid for seed in SEEDS]
    if index < 0 or index >= len(tasks):
        raise IndexError(f"Task index {index} outside 0..{len(tasks) - 1}")
    return tasks[index]


def validate_tuning_params(benchmark: str, method: str, params: dict) -> None:
    """Validate native tuning parameters."""
    mapped = "patch_feature" if benchmark == "patch" else "wsi_bag"
    _validate_common(mapped, method, params)
