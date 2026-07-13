from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    load_config,
    write_json,
)
from imbalance_benchmark.construction import (
    allocate_counts,
    build_manifest_hash,
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

logger = logging.getLogger(__name__)

__all__ = ["cmd_freeze"]

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
) -> dict[str, Any]:
    """Write one frozen condition manifest and report its realized statistics."""
    cond_df = pd.concat(selector_rows, ignore_index=True)
    path = data_dir / f"manifest_{name}.csv"
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
) -> dict[str, Any]:
    """Construct balanced/moderate/severe conditions sharing one master ordering."""
    counts = _class_support_counts(train_df, classes, is_mil)
    available = [counts[c] for c in classes]
    selector = select_slides_round_robin if is_mil else select_patches_round_robin
    conditions = {}
    for name, rho in CONDITION_RHOS.items():
        allocated = allocate_counts(available, shared_t, rho, min_support)
        rows = [
            selector(
                cast(pd.DataFrame, train_df[train_df["cancer_type"] == cls]),
                allocated[idx],
                seed=seed,
            )
            for idx, cls in enumerate(classes)
        ]
        conditions[name] = _write_condition(
            name, dict(zip(classes, allocated)), rows, train_df, is_mil, seed, data_dir
        )
    return conditions


def _load_pilot_floor(
    pilot_report_path: Path, is_mil: bool, counts: dict[str, int]
) -> tuple[int, int, bool]:
    """Read the pilot's floor and exclusion status, capped to what's actually available."""
    requested = _min_support_from_pilot(pilot_report_path, is_mil)
    min_support = min(requested, min(counts.values()))
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
    args: argparse.Namespace,
) -> tuple[dict[str, Path], pd.DataFrame, bool, list[str], dict[str, int]]:
    """Load the training manifest and derive the regime, classes, and support counts."""
    config = load_config(args.config)
    paths = ensure_dirs(config)
    df = pd.read_csv(paths["data"] / "manifest.csv")
    train_df = cast(pd.DataFrame, df[df["split"] == "train"])
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    classes = sorted(train_df["cancer_type"].unique())
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
    return {
        "shared_T": shared_t,
        "min_support": min_support,
        "requested_min_support": requested_min_support,
        "excluded": excluded,
        "construction_seed": construction_seed,
        "conditions": _build_conditions(
            train_df,
            classes,
            shared_t,
            min_support,
            is_mil,
            construction_seed,
            paths["data"],
        ),
        "tail_assignments": build_tail_assignments(
            classes, derive_seed(args.seed, "assignment"), ordinal=False
        ),
        "natural": _write_natural_condition(train_df, paths["data"]),
    }


def cmd_freeze(args: argparse.Namespace) -> None:
    """Freeze the definitive condition manifests and content-hashed analysis manifest."""
    paths, train_df, is_mil, classes, counts = _load_train_context(args)
    min_support, requested_min_support, excluded = _load_pilot_floor(
        paths["data"] / "pilot_report.json", is_mil, counts
    )
    shared_t = len(classes) * min(counts[c] for c in classes)
    meta = _freeze_meta(
        args,
        paths,
        train_df,
        is_mil,
        classes,
        shared_t,
        min_support,
        requested_min_support,
        excluded,
    )
    write_json(paths["data"] / "manifest_freeze.json", meta)
