from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast, Iterator

import pandas as pd

from imbalance_benchmark.analysis.inference.preflight import (
    bootstrap_preflight,
    require_valid_preflight,
)
from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    load_config,
    sign_file,
    split_paths,
    verify_signed_file,
    write_json,
)
from imbalance_benchmark.manifest.construction_helpers import (
    cap_feasible_shared_total,
    class_support_counts,
)
from imbalance_benchmark.manifest.freeze import (
    build_tail_assignments,
    lock_manifest_freeze,
)
from imbalance_benchmark.manifest.freezing import _freeze_meta, _pilot_constraints
from imbalance_benchmark.construction import locked_class_names
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.datasets.data import slide_level_identity
from imbalance_benchmark.datasets.features.provenance_lock import (
    attach_frozen_provenance,
    verify_prepared_feature_provenance,
)

logger = logging.getLogger(__name__)


@contextmanager
def _phase(name: str) -> Iterator[None]:
    """Log a freeze phase's start and duration so a stuck run is visible live."""
    start = time.perf_counter()
    logger.info("freeze: %s starting", name)
    yield
    logger.info("freeze: %s done in %.1fs", name, time.perf_counter() - start)


def _load_pilot_floor(
    pilot_report_path: Path, is_mil: bool, counts: dict[str, int]
) -> tuple[int, int, bool, int]:
    constraints = _pilot_constraints(pilot_report_path, is_mil)
    requested = constraints.patch_floor
    min_support = max(requested, 20) if not is_mil else requested
    excluded = (
        json.loads(pilot_report_path.read_text()).get("excluded", False)
        if pilot_report_path.exists()
        else False
    )
    if excluded:
        logger.warning(
            "Pilot marked this dataset-regime excluded (insufficient support "
            "even for the balanced condition); freezing anyway for inspection only."
        )
    if min(counts.values()) < min_support:
        excluded = True
    return min_support, requested, excluded, constraints.independent_floor


def _load_train_context(
    args: argparse.Namespace, paths: dict[str, Path]
) -> tuple[dict[str, Path], pd.DataFrame, bool, list[str], dict[str, int]]:
    config = load_config(args.config)
    df = pd.read_csv(paths["data"] / "manifest.csv")
    train_df = cast(pd.DataFrame, df[df["split"] == "train"])
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    classes = locked_class_names(df)
    counts = class_support_counts(train_df, is_mil)
    return paths, train_df, is_mil, classes, counts


def _load_split_context(
    args: argparse.Namespace, paths: dict[str, Path]
) -> tuple[dict[str, Path], pd.DataFrame, bool, list[str], int, int, bool, int] | None:
    paths, train_df, is_mil, classes, counts = _load_train_context(args, paths)
    pilot_path = paths["data"] / "pilot_report.json"
    if not pilot_path.exists():
        raise RuntimeError(
            "A signed pilot_report.json is required before definitive freeze"
        )
    # The pilot floor determines the frozen support; its evidence must be the
    # signed pilot, not a file that was edited after signing.
    verify_signed_file(pilot_path)
    min_sup, req_sup, excluded, independent_floor = _load_pilot_floor(
        pilot_path, is_mil, counts
    )
    if excluded:
        _write_exclusion(paths, min_sup, req_sup)
        return None
    return (
        paths,
        train_df,
        is_mil,
        classes,
        min_sup,
        req_sup,
        excluded,
        independent_floor,
    )


def freeze_split(args: argparse.Namespace) -> None:
    """Freeze one patient split's manifests, provenance, and label-only preflight."""
    logger.info("freeze: split %s starting", args.split_index)
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    feature_provenance = verify_prepared_feature_provenance(config, paths["data"])
    ctx = _load_split_context(args, paths)
    if ctx is None:
        return
    with _phase("condition and tail-assignment construction"):
        meta = _freeze_metadata(args, *ctx)
    with _phase("bootstrap preflight"):
        _attach_preflight(meta, paths, config, args.seed)
    _attach_provenance(meta, paths, config, feature_provenance)
    _write_freeze_file(meta, paths["data"] / "manifest_freeze.json")
    logger.info("freeze: split %s complete", args.split_index)


def _freeze_metadata(
    args: argparse.Namespace,
    paths: dict[str, Path],
    train_df: pd.DataFrame,
    is_mil: bool,
    classes: list[str],
    minimum: int,
    requested: int,
    excluded: bool,
    independent_floor: int,
) -> dict[str, Any]:
    """Build definitive metadata after the split has passed pilot eligibility."""
    config = load_config(args.config)
    assignments = build_tail_assignments(
        classes,
        derive_seed(args.seed, "assignment"),
        ordinal=str(config.get("dataset", {}).get("name", "")) == "panda"
        and config.get("dataset", {}).get("regime", "patch") == "wsi",
    )
    total = cap_feasible_shared_total(
        train_df,
        classes,
        minimum,
        is_mil,
        derive_seed(args.seed, "definitive_construction"),
        independent_floor,
        assignments,
    )
    return _freeze_meta(
        args,
        paths,
        train_df,
        is_mil,
        classes,
        total,
        minimum,
        requested,
        excluded,
        independent_floor,
        assignments,
    )


def _attach_provenance(
    meta: dict[str, Any],
    paths: dict[str, Path],
    config: dict[str, Any],
    feature_provenance: dict[str, str] | None = None,
) -> None:
    """Attach prepared dataset and feature provenance to a freeze."""
    attach_frozen_provenance(meta, paths["data"], config, feature_provenance)


def _write_freeze_file(meta: dict[str, Any], freeze_path: Path) -> None:
    """Persist and sign the locked freeze manifest."""
    meta["path"] = str(freeze_path)
    write_json(freeze_path, lock_manifest_freeze(meta))
    sign_file(freeze_path)


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


def _attach_preflight(
    meta: dict[str, Any],
    paths: dict[str, Path],
    config: dict[str, Any],
    seed: int,
) -> None:
    """Persist and content-hash the shared label-only bootstrap preflight."""
    path = paths["data"] / "bootstrap_preflight.json"
    preflight = _preflight(config, seed)
    require_valid_preflight(preflight)
    write_json(path, preflight)
    meta["bootstrap_preflight"] = {
        "path": str(path),
        "sha256": compute_sha256(path),
        "is_descriptive_only": preflight["is_descriptive_only"],
    }


def _preflight(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Build the joint test identity required by the frozen bootstrap preflight."""
    frames = []
    for index in range(3):
        manifest = split_paths(ensure_dirs(config), index)["data"] / "manifest.csv"
        if manifest.exists():
            frame = pd.read_csv(manifest)
            test = cast(pd.DataFrame, frame[frame["split"] == "test"])
            if config.get("dataset", {}).get("regime", "patch") == "wsi":
                test = slide_level_identity(test)
            frames.append(test.assign(patient_split=index))
    if len(frames) != 3:
        raise RuntimeError(
            "Exactly three prepared patient splits are required before a definitive freeze"
        )
    identity = pd.concat(frames, ignore_index=True)
    n_replicates = int(config.get("analysis", {}).get("bootstrap_replicates", 10_000))
    logger.info(
        "freeze: bootstrapping %d replicates over %d test rows",
        n_replicates,
        len(identity),
    )
    return bootstrap_preflight(identity, n_replicates, derive_seed(seed, "resampling"))
