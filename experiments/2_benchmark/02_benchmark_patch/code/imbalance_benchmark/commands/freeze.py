from __future__ import annotations

import argparse
import json
import logging

from imbalance_benchmark.commands.freeze_execution import (
    _attach_preflight,
    freeze_split,
)
from imbalance_benchmark.manifest.freezing import _build_conditions
from imbalance_benchmark.analysis.predictors.signals.signal_profile import (
    write_signal_profile,
)
from imbalance_benchmark.analysis.inference.crossed_permutation import load_freeze
from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    load_config,
    sign_file,
    split_paths,
    write_json,
)
from imbalance_benchmark.manifest.freeze import (
    lock_manifest_freeze,
    verify_manifest_freeze,
)
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.modeling.context import get_grid_configs, roster_for_regime

__all__ = [
    "cmd_freeze",
    "cmd_signals",
    "cmd_amend_grids",
    "cmd_refreeze_preflight",
    "_build_conditions",
]

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


def _amended_freeze_meta(meta: dict, is_mil: bool, old_file_hash: str) -> dict:
    """Add missing roster methods' grids to a frozen manifest, chaining its lock."""
    new_grids = {
        method: get_grid_configs(method, len(meta["class_names"]))
        for method in roster_for_regime(is_mil)
    }
    for method, grid in meta["method_grids"].items():
        if new_grids.get(method) != grid:
            raise RuntimeError(
                f"amend-grids refuses to change existing method '{method}'s grid; "
                "only additions to the roster are permitted"
            )
    amended = dict(meta)
    amended["method_grids"] = new_grids
    amended["supersedes"] = [meta["content_sha256"], *meta.get("supersedes", [])]
    amended["superseded_freeze_file_hashes"] = [
        old_file_hash,
        *meta.get("superseded_freeze_file_hashes", []),
    ]
    amended.pop("content_sha256", None)
    return amended


def cmd_amend_grids(args: argparse.Namespace) -> None:
    """Add newly rostered methods' grids to a frozen manifest without a full re-freeze.

    ``method_grids`` is a pure function of the roster and class count and touches
    no manifest CSV, split, or training set, so it is amended in place with a
    ``content_sha256`` supersession chain instead of a freeze rebuild that would
    invalidate every existing run, tuning shard, and signal profile.
    """
    if args.split_index is None:
        for index in range(3):
            cmd_amend_grids(argparse.Namespace(**{**vars(args), "split_index": index}))
        return
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    freeze_path = paths["data"] / "manifest_freeze.json"
    meta = json.loads(freeze_path.read_text())
    verify_manifest_freeze(meta)
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    amended = _amended_freeze_meta(meta, is_mil, compute_sha256(freeze_path))
    write_json(freeze_path, lock_manifest_freeze(amended))
    sign_file(freeze_path)
    added = len(amended["method_grids"]) - len(meta["method_grids"])
    logger.info("amend-grids: split %s added %d method(s)", args.split_index, added)


def _chain_supersession(meta: dict, old_content_hash: str, old_file_hash: str) -> None:
    """Record a prior freeze's hashes so `accepted_freeze_hashes` still accepts old runs."""
    meta["supersedes"] = [old_content_hash, *meta.get("supersedes", [])]
    meta["superseded_freeze_file_hashes"] = [
        old_file_hash,
        *meta.get("superseded_freeze_file_hashes", []),
    ]
    meta.pop("content_sha256", None)


def cmd_refreeze_preflight(args: argparse.Namespace) -> None:
    """Recompute only the label-only bootstrap preflight against an existing signed freeze.

    Fallback for when the preflight estimator changes but construction did
    not: a full re-freeze would drop any supersession chain a prior
    ``amend-grids`` already attached, so ``analyze`` would reject every
    existing confirm run via :func:`accepted_freeze_hashes`. This chains the
    prior content hash the same way ``amend-grids`` does instead.
    """
    if args.split_index is None:
        for index in range(3):
            cmd_refreeze_preflight(
                argparse.Namespace(**{**vars(args), "split_index": index})
            )
        return
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    freeze_path = paths["data"] / "manifest_freeze.json"
    meta = json.loads(freeze_path.read_text())
    verify_manifest_freeze(meta)
    old_file_hash = compute_sha256(freeze_path)
    _attach_preflight(meta, paths, config, args.seed)
    _chain_supersession(meta, meta["content_sha256"], old_file_hash)
    write_json(freeze_path, lock_manifest_freeze(meta))
    sign_file(freeze_path)
    logger.info("refreeze-preflight: split %s done", args.split_index)
