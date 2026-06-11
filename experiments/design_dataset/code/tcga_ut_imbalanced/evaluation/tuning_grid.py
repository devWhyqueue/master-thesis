"""Validation-tuning grids aligned with the class-imbalance benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

SEEDS = (0, 1, 2)
CLASS_ORDERS = ("native_prevalence", "easy_to_difficult", "difficult_to_easy")
LAMBDAS = (0.0, 1.0, 1.3)

PATCH_SPECS = (
    ("weighted_ce", "weight_power", [0.25, 0.5, 0.75, 1.0]),
    ("focal", "focal_gamma", [0.5, 1.0, 1.5, 2.0]),
    ("balanced_sampler", "sampler_power", [0.5, 0.75, 1.0]),
    ("ce_soft_f1", "metric_loss_weight", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
    ("ce_soft_mcc", "metric_loss_weight", [0.25, 0.5, 1.0, 2.0]),
    ("cfal", "cfal_gamma", [0.5, 1.0, 2.0, 5.0]),
    ("divide_conquer", "dnc_k_clusters", [5.0, 10.0, 15.0, 20.0]),
)

WSI_SPECS = (
    ("mil_weighted_ce", "weight_power", [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]),
    ("mil_focal", "focal_gamma", [0.5, 1.0, 1.5, 2.0]),
    ("mil_balanced_sampler_ce", "sampler_power", [0.5, 0.75, 1.0]),
    ("rankmix_mil", "rankmix_alpha", [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
    ("sc_mil", "sc_mil_temperature", [0.05, 0.1, 0.2, 0.5]),
    ("mde_mil", "mde_mil_consistency_weight", [0.1, 0.25, 0.3, 0.5, 1.0, 2.0, 4.0]),
)


@dataclass(frozen=True)
class Regime:
    """One constructed training-distribution regime."""

    class_order_name: str
    parameter: float

    @property
    def label(self) -> str:
        """Return a stable directory label for one regime."""
        return f"order={self.class_order_name}/param={self.parameter:g}"


@dataclass(frozen=True)
class TuningVariant:
    """One hyperparameter configuration for validation tuning."""

    benchmark: str
    method: str
    variant: str
    params: dict[str, float]

    @property
    def params_json(self) -> str:
        """Return compact JSON for CLI forwarding."""
        return json.dumps(self.params, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class TuningTask:
    """One tuning run: regime, method variant, and seed."""

    regime: Regime
    variant: TuningVariant
    seed: int


def regimes() -> list[Regime]:
    """Return all class-order and lambda regimes."""
    return [Regime(order, parameter) for order in CLASS_ORDERS for parameter in LAMBDAS]


def patch_grid() -> list[TuningVariant]:
    """Return the patch validation-tuning grid."""
    return _grid_from_specs("patch", PATCH_SPECS)


def wsi_grid() -> list[TuningVariant]:
    """Return the WSI validation-tuning grid."""
    return _grid_from_specs("wsi", WSI_SPECS)


def patch_tasks() -> list[TuningTask]:
    """Return all patch tuning tasks across regimes and seeds."""
    return _tasks_for_grid(patch_grid())


def wsi_tasks() -> list[TuningTask]:
    """Return all WSI tuning tasks across regimes and seeds."""
    return _tasks_for_grid(wsi_grid())


def task_for_index(benchmark: str, index: int) -> TuningTask:
    """Map a flat array index to one tuning task."""
    tasks = patch_tasks() if benchmark == "patch" else wsi_tasks()
    if index < 0 or index >= len(tasks):
        raise IndexError(f"Task index {index} outside 0..{len(tasks) - 1}")
    return tasks[index]


def task_count(benchmark: str) -> int:
    """Return the number of array tasks for one benchmark."""
    grid = patch_grid() if benchmark == "patch" else wsi_grid()
    return len(grid) * len(SEEDS) * len(regimes())


def validate_tuning_params(benchmark: str, method: str, params: dict[str, Any]) -> None:
    """Fail if a trainer receives unsupported tuning parameters."""
    allowed = {
        (variant.benchmark, variant.method, next(iter(variant.params)))
        for variant in patch_grid() + wsi_grid()
    }
    for key in params:
        if (benchmark, method, key) not in allowed:
            raise ValueError(f"Unsupported tuning parameter for {method}: {key}")


def _tasks_for_grid(grid: list[TuningVariant]) -> list[TuningTask]:
    tasks: list[TuningTask] = []
    for regime in regimes():
        for variant in grid:
            for seed in SEEDS:
                tasks.append(TuningTask(regime, variant, seed))
    return tasks


def _grid_from_specs(
    benchmark: str, specs: tuple[tuple[str, str, list[float]], ...]
) -> list[TuningVariant]:
    grid: list[TuningVariant] = []
    for method, key, values in specs:
        for value in values:
            grid.append(
                TuningVariant(benchmark, method, f"{key}={value:g}", {key: value})
            )
    return grid
