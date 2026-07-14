from __future__ import annotations
import argparse
import logging
import json
from pathlib import Path
from typing import Any, cast
import pandas as pd
from imbalance_benchmark.common import (
    compute_sha256,
    load_config,
)
from imbalance_benchmark.construction import (
    allocate_counts,
    build_manifest_hash,
    effective_rho,
    select_patches_round_robin,
    select_slides_round_robin,
)
from imbalance_benchmark.manifest.freeze import (
    achieved_rho,
    build_tail_assignments,
    contribution_stats,
    normalized_entropy,
)
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.manifest.seeds import SEED_ROLES
from imbalance_benchmark.modeling.context import get_grid_configs, roster_for_regime
from imbalance_benchmark.modeling.training import resolve_batch_size, update_budget
from imbalance_benchmark.manifest.construction_helpers import (
    class_construction_seed,
    class_support_counts,
    evidence_pool_hash,
    write_natural_condition,
)

CONDITION_RHOS = {"balanced": 1.0, "moderate": 10.0, "severe": 100.0}
logger = logging.getLogger(__name__)


def _min_support_from_pilot(pilot_report_path: Path, is_mil: bool) -> int:
    """Translate the pilot's independent-unit floor into a patch/slide-count floor.
    MIL support is already counted in slides, matching the pilot's unit. Patch
    support is counted in patches, so the patient/slide floor is converted via
    the largest pilot quota (patches held constant per contributing patient).
    """
    if not pilot_report_path.exists():
        return 10
    report = json.loads(pilot_report_path.read_text())
    definitive_floor = report["definitive_floor"]
    if is_mil:
        return definitive_floor
    quotas = [q for q in report["quotas"].values() if q is not None]
    return definitive_floor * (max(quotas) if quotas else 1)


def _write_condition(
    name: str,
    allocated: dict[str, int],
    selector_rows: list[pd.DataFrame],
    train_df: pd.DataFrame,
    is_mil: bool,
    seed: int,
    data_dir: Path,
    file_stem: str | None = None,
    evidence_pool_hash: str | None = None,
) -> dict[str, Any]:
    """Write one frozen condition manifest and report its realized statistics."""
    cond_df = pd.concat(selector_rows, ignore_index=True)
    path = data_dir / f"manifest_{file_stem or name}.csv"
    cond_df.to_csv(path, index=False)
    return {
        "path": str(path),
        "sha256": compute_sha256(path),
        "requested_rho": CONDITION_RHOS.get(name, 1.0),
        "achieved_rho": achieved_rho(allocated),
        "normalized_entropy": normalized_entropy(list(allocated.values())),
        "allocated_counts": allocated,
        "manifest_hash": build_manifest_hash(cond_df),
        "contribution_stats": contribution_stats(cond_df, train_df, is_mil),
        "construction_seed": seed,
        "evidence_pool_hash": evidence_pool_hash,
    }


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
) -> dict[str, Any]:
    """Construct cap-compliant controlled manifests from one fixed eligible pool."""
    counts = class_support_counts(train_df, is_mil)
    available = [counts[c] for c in classes]
    selector = select_slides_round_robin if is_mil else select_patches_round_robin
    pool_hash = evidence_pool_hash(train_df, classes, is_mil)
    allocations = {
        name: allocate_counts(
            available,
            shared_t,
            effective_rho(available, CONDITION_RHOS[name], min_support, shared_t),
            min_support,
        )
        for name in condition_names
    }
    conditions = {}
    for name in condition_names:
        allocated = allocations[name]
        rows = [
            selector(
                cast(pd.DataFrame, train_df[train_df["cancer_type"] == cls]),
                allocated[idx],
                # The class-specific stream fixes the eligible patient/slide
                # pool across conditions while each requested size is sampled
                # under its own 10%/5% caps (a prefix of a large capped sample
                # is not generally capped at the smaller size).
                seed=class_construction_seed(seed, cls),
            )
            for idx, cls in enumerate(classes)
        ]
        conditions[name] = _write_condition(
            name,
            dict(zip(classes, allocated)),
            rows,
            train_df,
            is_mil,
            seed,
            data_dir,
            f"{file_prefix}{name}",
            pool_hash,
        )
    return conditions


def _load_pilot_floor(
    pilot_report_path: Path, is_mil: bool, counts: dict[str, int]
) -> tuple[int, int, bool]:
    """Read the pilot's floor and exclusion status, capped to what's actually available."""
    requested = _min_support_from_pilot(pilot_report_path, is_mil)
    # With a strict 5% per-slide patch cap, a patch condition needs at least
    # 20 examples per class to admit even one patch from a slide.
    required_floor = max(requested, 20) if not is_mil else requested
    min_support = required_floor
    excluded = (
        json.loads(pilot_report_path.read_text()).get("excluded", False)
        if pilot_report_path.exists()
        else False
    )
    if excluded:
        logger.warning(
            "Pilot marked this dataset-regime excluded (insufficient independent "
            "units even for the balanced condition); freezing anyway for inspection "
            "but downstream analysis must treat it as excluded."
        )
    if min(counts.values()) < min_support:
        excluded = True
    return min_support, requested, excluded


def _load_train_context(
    args: argparse.Namespace, paths: dict[str, Path]
) -> tuple[dict[str, Path], pd.DataFrame, bool, list[str], dict[str, int]]:
    """Load the training manifest and derive the regime, classes, and support counts."""
    config = load_config(args.config)
    df = pd.read_csv(paths["data"] / "manifest.csv")
    train_df = cast(pd.DataFrame, df[df["split"] == "train"])
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    dataset_name = str(config.get("dataset", {}).get("name", ""))
    observed = train_df["cancer_type"].astype(str).unique().tolist()
    classes = observed
    if dataset_name == "panda" and all(name.startswith("ISUP") for name in observed):
        classes = sorted(observed, key=lambda name: int(name.removeprefix("ISUP")))
    counts = class_support_counts(train_df, is_mil)
    if dataset_name != "panda" or not all(name.startswith("ISUP") for name in observed):
        classes = sorted(observed, key=lambda name: (-counts[name], name))
        counts = class_support_counts(train_df, is_mil)
    return paths, train_df, is_mil, classes, counts


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
) -> dict[str, Any]:
    """Assemble the frozen analysis manifest: conditions, tail assignments, and provenance."""
    construction_seed = derive_seed(args.seed, "definitive_construction")
    config = load_config(args.config)
    assignments = build_tail_assignments(
        classes,
        derive_seed(args.seed, "assignment"),
        ordinal=str(config.get("dataset", {}).get("name", "")) == "panda"
        and bool(config.get("dataset", {}).get("regime", "patch") == "wsi"),
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
    )
    return {
        "shared_T": shared_t,
        "min_support": min_support,
        "requested_min_support": requested_min_support,
        "excluded": excluded,
        "construction_seed": construction_seed,
        "seed_roles": {role: derive_seed(args.seed, role) for role in SEED_ROLES},
        "method_grids": {
            method: get_grid_configs(method, len(classes))
            for method in roster_for_regime(is_mil)
        },
        "update_budgets": {
            "controlled": update_budget(shared_t, resolve_batch_size(config, is_mil)),
            "natural": update_budget(len(train_df), resolve_batch_size(config, is_mil)),
        },
        "conditions": native_conditions,
        "assignment_conditions": assignment_conditions,
        "tail_assignments": assignments,
        "natural": write_natural_condition(train_df, paths["data"]),
    }
