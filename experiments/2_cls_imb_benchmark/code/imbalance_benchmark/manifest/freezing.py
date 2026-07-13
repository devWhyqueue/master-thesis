from __future__ import annotations
import argparse
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

CONDITION_RHOS = {"balanced": 1.0, "moderate": 10.0, "severe": 100.0}


def _class_support_counts(
    train_df: pd.DataFrame, classes: list[str], is_mil: bool
) -> dict[str, int]:
    """Return each class's available allocation pool: unique slides for MIL, patch rows otherwise."""
    if is_mil:
        return train_df.groupby("cancer_type")["slide_id"].nunique().to_dict()
    return train_df["cancer_type"].value_counts().to_dict()


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
    counts = _class_support_counts(train_df, classes, is_mil)
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
                seed=derive_seed(seed, "definitive_construction") ^ idx,
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


def _write_natural_condition(train_df: pd.DataFrame, data_dir: Path) -> dict[str, Any]:
    """Write the descriptive full-training-set anchor, excluded from deficit estimands."""
    path = data_dir / "manifest_natural.csv"
    train_df.to_csv(path, index=False)
    return {
        "path": str(path),
        "sha256": compute_sha256(path),
        "note": "descriptive anchor; excluded from imbalance deficit/recovery estimands",
    }


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
    # PANDA ISUP is clinical ordinal order; other tasks retain native support
    # order (with lexical ties only for determinism), never alphabetical order.
    if dataset_name == "panda" and all(name.startswith("ISUP") for name in observed):
        classes = sorted(observed, key=lambda name: int(name.removeprefix("ISUP")))
    else:
        classes = sorted(
            observed,
            key=lambda name: (
                -int(train_df[train_df["cancer_type"] == name].shape[0]),
                name,
            ),
        )
    counts = _class_support_counts(train_df, classes, is_mil)
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
        "conditions": native_conditions,
        "assignment_conditions": assignment_conditions,
        "tail_assignments": assignments,
        "natural": _write_natural_condition(train_df, paths["data"]),
    }
