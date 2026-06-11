"""Validation-tuning grids shared across experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

SEEDS = (0, 1, 2)
PATCH_FEATURE_SPECS = (
    ("patch_feature_weighted_ce", "weight_power", [0.25, 0.5, 0.75, 1.0]),
    ("patch_feature_focal", "focal_gamma", [0.5, 1.0, 1.5, 2.0]),
    ("patch_feature_balanced_sampler_ce", "sampler_power", [0.5, 0.75, 1.0]),
    (
        "patch_feature_ce_soft_f1_balanced",
        "metric_loss_weight",
        [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    ),
    (
        "patch_feature_ce_soft_mcc_balanced",
        "metric_loss_weight",
        [0.25, 0.5, 1.0, 2.0],
    ),
    ("patch_feature_cfal", "cfal_gamma", [0.5, 1.0, 2.0, 5.0]),
    ("patch_feature_divide_conquer", "dnc_k_clusters", [5.0, 10.0, 15.0, 20.0]),
    ("patch_feature_progan_aug", "final_depth_epochs", [10.0, 25.0, 50.0]),
)
WSI_BAG_SPECS = (
    ("mil_weighted_ce", "weight_power", [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]),
    ("mil_focal", "focal_gamma", [0.5, 1.0, 1.5, 2.0]),
    ("mil_balanced_sampler_ce", "sampler_power", [0.5, 0.75, 1.0]),
    ("rankmix_mil", "rankmix_alpha", [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
    ("sc_mil", "sc_mil_temperature", [0.05, 0.1, 0.2, 0.5]),
    ("mde_mil", "mde_mil_consistency_weight", [0.1, 0.25, 0.3, 0.5, 1.0, 2.0, 4.0]),
)


@dataclass(frozen=True)
class TuningVariant:
    """One validation-tuning configuration."""

    benchmark: str
    method: str
    variant: str
    params: dict[str, float]

    @property
    def label(self) -> str:
        return ", ".join(f"{key}={value:g}" for key, value in self.params.items())

    @property
    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True, separators=(",", ":"))


def patch_feature_grid() -> list[TuningVariant]:
    return grid_from_specs("patch_feature", PATCH_FEATURE_SPECS)


def wsi_bag_grid() -> list[TuningVariant]:
    return grid_from_specs("wsi_bag", WSI_BAG_SPECS)


def grid_for_benchmark(benchmark: str) -> list[TuningVariant]:
    if benchmark == "patch_feature":
        return patch_feature_grid()
    if benchmark == "wsi_bag":
        return wsi_bag_grid()
    raise ValueError(f"Unsupported tuning benchmark: {benchmark}")


def task_for_array_index(benchmark: str, array_index: int) -> tuple[TuningVariant, int]:
    grid = grid_for_benchmark(benchmark)
    total = len(grid) * len(SEEDS)
    if array_index < 0 or array_index >= total:
        raise IndexError(f"Array index {array_index} outside 0..{total - 1}")
    return grid[array_index // len(SEEDS)], SEEDS[array_index % len(SEEDS)]


def task_count(benchmark: str) -> int:
    return len(grid_for_benchmark(benchmark)) * len(SEEDS)


def validate_tuning_params(benchmark: str, method: str, params: dict[str, Any]) -> None:
    allowed = {
        (variant.benchmark, variant.method, next(iter(variant.params)))
        for variant in patch_feature_grid() + wsi_bag_grid()
    }
    for key in params:
        if (benchmark, method, key) not in allowed:
            raise ValueError(f"Unsupported tuning parameter for {method}: {key}")


def grid_from_specs(
    benchmark: str, specs: tuple[tuple[str, str, list[float]], ...]
) -> list[TuningVariant]:
    grid: list[TuningVariant] = []
    for method, key, values in specs:
        for value in values:
            grid.append(
                TuningVariant(benchmark, method, f"{key}={value:g}", {key: value})
            )
    return grid
