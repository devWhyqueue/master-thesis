from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import pandas as pd
from imbalance_benchmark.common import load_config
from imbalance_benchmark.construction import (
    allocate_counts,
    effective_rho,
    select_patches_round_robin,
    select_slides_round_robin,
)
from imbalance_benchmark.manifest.freeze import build_tail_assignments, write_condition
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.manifest.seeds import SEED_ROLES
from imbalance_benchmark.modeling.context import get_grid_configs, roster_for_regime
from imbalance_benchmark.modeling.training import resolve_batch_size, update_budget
from imbalance_benchmark.manifest.construction_helpers import (
    CONDITION_RHOS,
    _retains_fixed_pool,
    assignment_allocations,
    class_construction_seed,
    class_support_counts,
    designate_shared_patch_pools,
    write_natural_condition,
)
from imbalance_benchmark.manifest.statistics import evidence_pool_hash


@dataclass(frozen=True)
class PilotConstraints:
    """Allocation and independent-unit floors frozen from the pilot."""

    patch_floor: int
    independent_floor: int


def _pilot_constraints(pilot_report_path: Path, is_mil: bool) -> PilotConstraints:
    """Translate the pilot's independent-unit floor into a patch/slide-count floor.
    MIL support is already counted in slides, matching the pilot's unit. Patch
    support is counted in patches, so the patient/slide floor is converted via
    the largest pilot quota (patches held constant per contributing patient).
    """
    if not pilot_report_path.exists():
        return PilotConstraints(10, 10)
    report = json.loads(pilot_report_path.read_text())
    definitive_floor = report["definitive_floor"]
    if is_mil:
        return PilotConstraints(definitive_floor, definitive_floor)
    quotas = [q for q in report["quotas"].values() if q is not None]
    return PilotConstraints(
        definitive_floor * (max(quotas) if quotas else 1), definitive_floor
    )


def _build_conditions(
    train_df: pd.DataFrame,
    classes: list[str],
    shared_t: int,
    min_support: int,
    is_mil: bool,
    seed: int,
    data_dir: Path,
    file_prefix: str = "",
    condition_names: tuple[str, ...] = tuple(CONDITION_RHOS),
    independent_floor: int | None = None,
    fixed_pools: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Construct cap-compliant controlled manifests from one fixed eligible pool."""
    counts = class_support_counts(train_df, is_mil)
    available = [counts[c] for c in classes]
    selector = select_slides_round_robin if is_mil else select_patches_round_robin
    allocations = {
        name: allocate_counts(
            available,
            shared_t,
            effective_rho(available, CONDITION_RHOS[name], min_support, shared_t),
            min_support,
        )
        for name in condition_names
    }
    fixed_pools = fixed_pools or (
        {
            cls: designate_shared_patch_pools(
                train_df,
                {
                    "current": {
                        name: dict(zip(classes, allocation, strict=True))
                        for name, allocation in allocations.items()
                    }
                },
                independent_floor or 10,
                seed,
            )[cls]
            for cls in classes
        }
        if not is_mil
        else {}
    )
    pool_df = (
        pd.concat(fixed_pools.values(), ignore_index=True) if fixed_pools else train_df
    )
    pool_hash = evidence_pool_hash(pool_df, classes, is_mil)
    conditions = {}
    for name in condition_names:
        allocated = allocations[name]
        rows = [
            selector(
                fixed_pools.get(
                    cls,
                    cast(pd.DataFrame, train_df[train_df["cancer_type"] == cls]),
                ),
                allocated[idx],
                # Each condition samples from this same explicit patient/slide
                # pool; its hash records the actual designated units.
                seed=class_construction_seed(seed, cls),
            )
            for idx, cls in enumerate(classes)
        ]
        if not is_mil and independent_floor is not None:
            for cls, selected in zip(classes, rows, strict=True):
                pool = fixed_pools[cls]
                if not _retains_fixed_pool(selected, pool):
                    raise ValueError(
                        "Controlled patch allocation does not retain its fixed evidence pool"
                    )
        conditions[name] = write_condition(
            name,
            dict(zip(classes, allocated)),
            rows,
            train_df,
            is_mil,
            seed,
            data_dir,
            f"{file_prefix}{name}",
            pool_hash,
            available,
            min_support,
        )
    return conditions


def _freeze_meta(
    args: argparse.Namespace,
    paths: dict[str, Path],
    train_df: pd.DataFrame,
    is_mil: bool,
    classes: list[str],
    shared_t: int,
    min_support: int,
    requested_min_support: int,
    excluded: bool,
    independent_floor: int,
    assignments: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Assemble the frozen analysis manifest: conditions, tail assignments, and provenance."""
    construction_seed = derive_seed(args.seed, "definitive_construction")
    config = load_config(args.config)
    assignments = assignments or build_tail_assignments(
        classes,
        derive_seed(args.seed, "assignment"),
        ordinal=str(config.get("dataset", {}).get("name", "")) == "panda"
        and bool(config.get("dataset", {}).get("regime", "patch") == "wsi"),
    )
    shared_pools = (
        designate_shared_patch_pools(
            train_df,
            assignment_allocations(train_df, assignments, shared_t, min_support),
            independent_floor,
            construction_seed,
        )
        if not is_mil
        else None
    )
    assignment_conditions = {
        assignment: _build_conditions(
            train_df,
            order,
            shared_t,
            min_support,
            is_mil,
            construction_seed,
            paths["data"],
            file_prefix=f"{assignment}_",
            condition_names=("moderate", "severe"),
            independent_floor=independent_floor,
            fixed_pools=shared_pools,
        )
        for assignment, order in assignments.items()
    }
    native_conditions = _build_conditions(
        train_df,
        classes,
        shared_t,
        min_support,
        is_mil,
        construction_seed,
        paths["data"],
        condition_names=("balanced",),
        independent_floor=independent_floor,
        fixed_pools=shared_pools,
    )
    return {
        "class_names": classes,
        "label_to_index": {name: index for index, name in enumerate(classes)},
        "shared_T": shared_t,
        "min_support": min_support,
        "requested_min_support": requested_min_support,
        "independent_floor": independent_floor,
        "excluded": excluded,
        "construction_seed": construction_seed,
        "seed_roles": {role: derive_seed(args.seed, role) for role in SEED_ROLES},
        "method_grids": {
            method: get_grid_configs(method, len(classes))
            for method in roster_for_regime(is_mil)
        },
        "update_budgets": {
            "controlled": update_budget(shared_t, resolve_batch_size(config, is_mil)),
            "natural": update_budget(
                sum(class_support_counts(train_df, is_mil).values()),
                resolve_batch_size(config, is_mil),
            ),
        },
        "runtime_config": config,
        "conditions": native_conditions,
        "assignment_conditions": assignment_conditions,
        "tail_assignments": assignments,
        "natural": write_natural_condition(train_df, paths["data"], is_mil),
    }
