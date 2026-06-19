"""Validation-tuning grids aligned with the class-imbalance benchmark."""

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

CLASS_ORDERS = ("native_prevalence",)
LAMBDAS = (0.8, 1.1, 1.3)

PATCH_SPECS = PATCH_FEATURE_SPECS
WSI_SPECS = WSI_BAG_SPECS


@dataclass(frozen=True)
class Regime:
    class_order_name: str
    parameter: float

    @property
    def label(self) -> str:
        """Return the directory-style regime label."""
        return f"order={self.class_order_name}/param={self.parameter:g}"


@dataclass(frozen=True)
class TuningTask:
    regime: Regime
    variant: TuningVariant
    seed: int


def regimes() -> list[Regime]:
    """Return all (class_order, lambda) regime combinations."""
    return [Regime(order, parameter) for order in CLASS_ORDERS for parameter in LAMBDAS]


def patch_grid() -> list[TuningVariant]:
    """Return the full patch-feature tuning grid."""
    return grid_from_specs("patch_feature", PATCH_SPECS)


def wsi_grid() -> list[TuningVariant]:
    """Return the full WSI-bag tuning grid."""
    return grid_from_specs("wsi_bag", WSI_SPECS)


def patch_tasks() -> list[TuningTask]:
    """Return all patch-feature tuning tasks (regime x variant x seed)."""
    return _tasks_for_grid(patch_grid())


def wsi_tasks() -> list[TuningTask]:
    """Return all WSI-bag tuning tasks (regime x variant x seed)."""
    return _tasks_for_grid(wsi_grid())


def task_for_index(benchmark: str, index: int) -> TuningTask:
    """Return the tuning task at the given flat index for the named benchmark."""
    tasks = patch_tasks() if benchmark == "patch" else wsi_tasks()
    if index < 0 or index >= len(tasks):
        raise IndexError(f"Task index {index} outside 0..{len(tasks) - 1}")
    return tasks[index]


def task_count(benchmark: str) -> int:
    """Return the total number of tuning tasks for the named benchmark."""
    grid = patch_grid() if benchmark == "patch" else wsi_grid()
    return len(grid) * len(SEEDS) * len(regimes())


def validate_tuning_params(benchmark: str, method: str, params: dict) -> None:
    """Delegate to the shared tuning-parameter validator."""
    mapped = "patch_feature" if benchmark == "patch" else "wsi_bag"
    _validate_common(mapped, method, params)


def _tasks_for_grid(grid: list[TuningVariant]) -> list[TuningTask]:
    tasks: list[TuningTask] = []
    for regime in regimes():
        for variant in grid:
            for seed in SEEDS:
                tasks.append(TuningTask(regime, variant, seed))
    return tasks
