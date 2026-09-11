"""Census, per-shard fit, and allocation stages for the site-coverage experiment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.common import (
    output_root,
    sign_file,
    verify_signed_file,
    write_json,
)

from breadth import exp2_split_paths
from breadth.fit import fit_and_record, init_shard

from sites import (
    ALLOCATIONS,
    FIT_SHARD_COUNT,
    MIN_SITE_CLASSES,
    N_DRAWS,
    N_SPLITS,
    allocation_dir,
)
from sites.allocation import draw_allocations, site_class_names

__all__ = ["decode_shard_index", "load_census", "run_census", "run_fit_shard"]

logger = logging.getLogger(__name__)


def decode_shard_index(shard_index: int) -> tuple[int, str]:
    """Decode a shard index into (split_index, allocation) over 3 splits x 3 allocations."""
    if shard_index not in range(FIT_SHARD_COUNT):
        raise ValueError(f"shard_index must be in [0, {FIT_SHARD_COUNT - 1}]")
    names = list(ALLOCATIONS)
    s_idx = shard_index // len(names)
    a_idx = shard_index % len(names)
    return s_idx, names[a_idx]


def _train_manifest(config: dict[str, Any], split_idx: int) -> pd.DataFrame:
    exp2_p = exp2_split_paths(config, split_idx)
    manifest = pd.read_csv(exp2_p["data"] / "manifest.csv")
    return manifest.query("split == 'train'").reset_index(drop=True)


def load_census(config: dict[str, Any]) -> dict[str, Any]:
    """Load and verify the signed site-class census."""
    census_p = output_root(config) / "data" / "census.json"
    verify_signed_file(census_p)
    return json.loads(census_p.read_text(encoding="utf-8"))


def run_census(config: dict[str, Any]) -> Path:
    """Census the site-class pool per split; write and sign data/census.json.

    Writes before gating so a failing census is still inspectable, and raises
    afterwards so the downstream ``afterok`` fit array never starts.
    """
    class_names = list(load_freeze_meta(exp2_split_paths(config, 0))["class_names"])
    train_dfs = {s: _train_manifest(config, s) for s in range(N_SPLITS)}
    site_classes = sorted(site_class_names(train_dfs, class_names))
    payload = {
        "site_classes": site_classes,
        "n_site_classes": len(site_classes),
        "min_required": MIN_SITE_CLASSES,
    }
    out_p = output_root(config) / "data" / "census.json"
    write_json(out_p, payload)
    sign_file(out_p)
    if len(site_classes) < MIN_SITE_CLASSES:
        raise RuntimeError(
            f"Only {len(site_classes)} site classes qualify (need >= "
            f"{MIN_SITE_CLASSES}); redesign the site levels before fitting."
        )
    return out_p


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Execute all draws of one (split, allocation) shard."""
    split_idx, allocation = decode_shard_index(shard_index)
    train_df, class_names, evals, paths = init_shard(config, split_idx)
    site_classes = load_census(config)["site_classes"]
    g, m = ALLOCATIONS[allocation]
    for draw_idx in range(N_DRAWS):
        logger.info(
            "Fitting split %d, allocation %s, draw %d", split_idx, allocation, draw_idx
        )
        frames, site_record = draw_allocations(
            train_df, class_names, site_classes, split_idx, draw_idx
        )
        result_dir = allocation_dir(paths, allocation, draw_idx)
        fit_and_record(
            config, result_dir, frames[allocation], class_names, (g, m, draw_idx), evals
        )
        write_json(result_dir / "sites.json", site_record)
