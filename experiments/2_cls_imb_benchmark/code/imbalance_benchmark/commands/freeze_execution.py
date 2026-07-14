from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.analysis.inference.preflight import bootstrap_preflight
from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    load_config,
    sign_file,
    split_paths,
    write_json,
)
from imbalance_benchmark.manifest.freezing import (
    _freeze_meta,
    _load_pilot_floor,
    _load_train_context,
)
from imbalance_benchmark.manifest.construction_helpers import cap_feasible_shared_total
from imbalance_benchmark.manifest.freeze import lock_manifest_freeze
from imbalance_benchmark.manifest.seeds import derive_seed


def freeze_split(args: argparse.Namespace) -> None:
    """Freeze one patient split's manifests, provenance, and label-only preflight."""
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    paths, train_df, is_mil, classes, counts = _load_train_context(args, paths)
    min_sup, req_sup, excluded = _load_pilot_floor(
        paths["data"] / "pilot_report.json", is_mil, counts
    )
    if excluded:
        _write_exclusion(paths, min_sup, req_sup)
        return
    meta = _freeze_metadata(
        args, paths, train_df, is_mil, classes, min_sup, req_sup, excluded
    )
    _attach_preflight(meta, paths, config, args.seed)
    freeze_path = paths["data"] / "manifest_freeze.json"
    meta["path"] = str(freeze_path)
    write_json(freeze_path, lock_manifest_freeze(meta))
    sign_file(freeze_path)


def _freeze_metadata(
    args: argparse.Namespace,
    paths: dict[str, Path],
    train_df: pd.DataFrame,
    is_mil: bool,
    classes: list[str],
    minimum: int,
    requested: int,
    excluded: bool,
) -> dict[str, Any]:
    """Build definitive metadata after the split has passed pilot eligibility."""
    total = cap_feasible_shared_total(
        train_df,
        classes,
        minimum,
        is_mil,
        derive_seed(args.seed, "definitive_construction"),
    )
    return _freeze_meta(
        args, paths, train_df, is_mil, classes, total, minimum, requested, excluded
    )


def _attach_preflight(
    meta: dict[str, Any],
    paths: dict[str, Path],
    config: dict[str, Any],
    seed: int,
) -> None:
    """Persist and content-hash the shared label-only bootstrap preflight."""
    path = paths["data"] / "bootstrap_preflight.json"
    preflight = _preflight(config, seed)
    write_json(path, preflight)
    meta["bootstrap_preflight"] = {
        "path": str(path),
        "sha256": compute_sha256(path),
        "is_descriptive_only": preflight["is_descriptive_only"],
    }


def _write_exclusion(paths: dict[str, Path], minimum: int, requested: int) -> None:
    """Record an explicit confirmation exclusion for an infeasible split."""
    write_json(
        paths["data"] / "confirmatory_exclusion.json",
        {
            "excluded": True,
            "reason": "independent-support floor or patch contribution caps not met",
            "min_support": minimum,
            "requested_min_support": requested,
        },
    )


def _preflight(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Build the joint test identity required by the frozen bootstrap preflight."""
    frames = []
    for index in range(3):
        manifest = split_paths(ensure_dirs(config), index)["data"] / "manifest.csv"
        if manifest.exists():
            frame = pd.read_csv(manifest)
            frames.append(
                cast(pd.DataFrame, frame[frame["split"] == "test"]).assign(
                    patient_split=index
                )
            )
    if len(frames) != 3:
        raise RuntimeError(
            "Exactly three prepared patient splits are required before a definitive freeze"
        )
    return bootstrap_preflight(
        pd.concat(frames, ignore_index=True),
        int(config.get("analysis", {}).get("bootstrap_replicates", 10_000)),
        derive_seed(seed, "resampling"),
    )
