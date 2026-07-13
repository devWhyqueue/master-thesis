from __future__ import annotations

import argparse
import logging
from typing import cast

import pandas as pd

from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    load_config,
    split_paths,
    write_json,
)
from imbalance_benchmark.analysis.inference.bootstrap import bootstrap_preflight
from imbalance_benchmark.construction import (
    max_shared_total,
)
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.manifest.freezing import (
    _freeze_meta,
    _load_pilot_floor,
    _load_train_context,
)

logger = logging.getLogger(__name__)

__all__ = ["cmd_freeze"]


def cmd_freeze(args: argparse.Namespace) -> None:
    """Freeze the definitive condition manifests and content-hashed analysis manifest."""
    if args.split_index is None:
        for index in range(3):
            cmd_freeze(argparse.Namespace(**vars(args), split_index=index))
        return
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    paths, train_df, is_mil, classes, counts = _load_train_context(args, paths)
    min_support, requested_min_support, excluded = _load_pilot_floor(
        paths["data"] / "pilot_report.json", is_mil, counts
    )
    if excluded:
        logger.warning(
            "Skipping definitive freeze because the pilot excluded this dataset-regime"
        )
        write_json(
            paths["data"] / "confirmatory_exclusion.json",
            {
                "excluded": True,
                "reason": "independent-support floor or patch contribution caps not met",
                "min_support": min_support,
                "requested_min_support": requested_min_support,
            },
        )
        return
    shared_t = max_shared_total([counts[c] for c in classes], min_support)
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
    test_frames = []
    for index in range(3):
        split_manifest = (
            split_paths(ensure_dirs(config), index)["data"] / "manifest.csv"
        )
        if split_manifest.exists():
            test_frame = pd.read_csv(split_manifest)
            test_frames.append(
                cast(pd.DataFrame, test_frame[test_frame["split"] == "test"]).assign(
                    patient_split=index
                )
            )
    if len(test_frames) != 3:
        raise RuntimeError(
            "Exactly three prepared patient splits are required before a definitive freeze"
        )
    test_rows = pd.concat(test_frames, ignore_index=True)
    preflight = bootstrap_preflight(
        test_rows,
        int(config.get("analysis", {}).get("bootstrap_replicates", 10_000)),
        derive_seed(args.seed, "resampling"),
    )
    preflight_path = paths["data"] / "bootstrap_preflight.json"
    write_json(preflight_path, preflight)
    meta["bootstrap_preflight"] = {
        "path": str(preflight_path),
        "sha256": compute_sha256(preflight_path),
        "is_descriptive_only": preflight["is_descriptive_only"],
    }
    write_json(paths["data"] / "manifest_freeze.json", meta)
