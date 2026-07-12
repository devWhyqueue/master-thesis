"""Validation-tuning grids with local filtering of removed methods."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import Callable, cast

_grid_module = import_module("common_code.tuning.grid")

PATCH_FEATURE_SPECS = tuple(
    spec
    for spec in cast(
        tuple[tuple[str, str, tuple[float, ...]], ...],
        getattr(_grid_module, "PATCH_FEATURE_SPECS"),
    )
    if spec[0] != "patch_feature_divide_conquer"
)
SEEDS = cast(tuple[int, ...], getattr(_grid_module, "SEEDS"))
TuningVariant = cast(type, getattr(_grid_module, "TuningVariant"))
WSI_BAG_SPECS = cast(
    tuple[tuple[str, str, tuple[float, ...]], ...],
    getattr(_grid_module, "WSI_BAG_SPECS"),
)

_grid_for_benchmark = cast(
    Callable[[str], Iterable[object]], getattr(_grid_module, "grid_for_benchmark")
)
_validate_tuning_params = cast(
    Callable[[str, str, dict[str, float]], None],
    getattr(_grid_module, "validate_tuning_params"),
)


def _filtered_variants(variants: Iterable[object]) -> list[object]:
    return [
        variant
        for variant in variants
        if getattr(variant, "method", None) != "patch_feature_divide_conquer"
    ]


def patch_feature_grid() -> list[object]:
    """Return patch-feature tuning variants after local method filtering."""
    return _filtered_variants(_grid_for_benchmark("patch_feature"))


def wsi_bag_grid() -> list[object]:
    """Return the unchanged WSI-bag tuning variants."""
    return list(_grid_for_benchmark("wsi_bag"))


def grid_for_benchmark(benchmark: str) -> list[object]:
    """Return tuning variants for one benchmark."""
    if benchmark == "patch_feature":
        return patch_feature_grid()
    return wsi_bag_grid()


def task_count(benchmark: str) -> int:
    """Return the array-task count for one benchmark."""
    return len(grid_for_benchmark(benchmark)) * len(SEEDS)


def task_for_array_index(benchmark: str, array_index: int) -> tuple[object, int]:
    """Resolve one array index into a tuning variant and seed."""
    variants = grid_for_benchmark(benchmark)
    if array_index < 0 or array_index >= len(variants) * len(SEEDS):
        raise IndexError(f"array index {array_index} out of range for {benchmark}")
    variant_index, seed_index = divmod(array_index, len(SEEDS))
    return variants[variant_index], SEEDS[seed_index]


def validate_tuning_params(
    benchmark: str, method: str, payload: dict[str, float]
) -> None:
    """Validate one tuning payload against the active benchmark grid."""
    if benchmark == "patch_feature" and method == "patch_feature_divide_conquer":
        raise ValueError(f"Unsupported tuning parameter for removed method {method}")
    _validate_tuning_params(benchmark, method, payload)


__all__ = [
    "PATCH_FEATURE_SPECS",
    "SEEDS",
    "TuningVariant",
    "WSI_BAG_SPECS",
    "grid_for_benchmark",
    "patch_feature_grid",
    "task_count",
    "task_for_array_index",
    "validate_tuning_params",
    "wsi_bag_grid",
]
