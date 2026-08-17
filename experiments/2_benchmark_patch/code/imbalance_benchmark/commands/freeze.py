from __future__ import annotations

import argparse
import logging

from imbalance_benchmark.commands.freeze_execution import freeze_split
from imbalance_benchmark.manifest.freezing import _build_conditions
from imbalance_benchmark.analysis.predictors.signals.signal_profile import (
    write_signal_profile,
)
from imbalance_benchmark.analysis.inference.crossed_permutation import load_freeze
from imbalance_benchmark.common import ensure_dirs, load_config, split_paths
from imbalance_benchmark.manifest.seeds import derive_seed

__all__ = ["cmd_freeze", "cmd_signals", "_build_conditions"]

logger = logging.getLogger(__name__)


def cmd_freeze(args: argparse.Namespace) -> None:
    """Freeze the definitive condition manifests and content-hashed analysis manifest."""
    if args.split_index is None:
        for index in range(3):
            cmd_freeze(argparse.Namespace(**{**vars(args), "split_index": index}))
        return
    freeze_split(args)


def cmd_signals(args: argparse.Namespace) -> None:
    """Write one split's pre-outcome signal profile (shortage scores, descriptive ICC/N_eff)."""
    if args.split_index is None:
        for index in range(3):
            cmd_signals(argparse.Namespace(**{**vars(args), "split_index": index}))
        return
    logger.info("signals: split %s starting", args.split_index)
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    if (paths["data"] / "confirmatory_exclusion.json").exists():
        return
    freeze = load_freeze(paths)
    if not freeze:
        raise RuntimeError("A signed manifest_freeze.json is required before signals")
    path = write_signal_profile(paths, freeze, derive_seed(args.seed, "resampling"))
    logger.info("signals: split %s wrote %s", args.split_index, path)
