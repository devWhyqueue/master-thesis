"""PANDA-only materialization and feature-reduction commands."""

from __future__ import annotations

import argparse

from imbalance_benchmark.common import ensure_dirs, load_config
from imbalance_benchmark.commands.prepare import _eligible_image_manifest
from imbalance_benchmark.datasets.panda_materialize import (
    audit_canary,
    materialize,
    reduce_feature_inventory,
)


def cmd_prepare_extract_reduce(args: argparse.Namespace) -> None:
    """Serially sign complete PANDA shard features after cache validation."""
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, dict) or dataset_cfg.get("name") != "panda":
        raise ValueError("PANDA feature reduction requires the PANDA patch config")
    frame = _eligible_image_manifest(config)
    if frame is None:
        raise ValueError("PANDA feature reduction requires image-backed inventory")
    reduce_feature_inventory(
        config, frame, ensure_dirs(config)["data"] / "features" / "panda"
    )


def cmd_materialize_panda(args: argparse.Namespace) -> None:
    """Audit legacy PANDA tiles and publish immutable project shards."""
    config = load_config(args.config)
    if getattr(args, "canary", False):
        audit_canary(config)
    else:
        materialize(config)
