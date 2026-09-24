"""Fit stage: exp-39's seven paired arms (B/R10/R100/P10/P100/S10/S100) of one
(encoder, split, draw) shard, reusing exp-25/27's allocator/weighting and exp-5's
tuning/calibration machinery under a frozen local lambda grid (``transfer.LAMBDAS``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from imbalance_benchmark.common import (
    N_PATIENT_SPLITS,
    RUN_RECORD_NAME,
    ensure_dirs,
    split_paths,
)

from breadth.fit import EvalPartition
from breadth.sampling import load_eval_partition

from sites import allocation_dir

from prevalence import patients_per_class
from prevalence.fit import _shard_context

from transfer import ARMS, ENCODERS, FIT_SOURCE, MAIN_DRAWS
from transfer.fit.tuning import CANDIDATES_NAME, fit_arm, tune_and_fit_draw
from transfer.schedule import load_train_identity

__all__ = [
    "CANDIDATES_NAME",
    "EvalPartition",
    "tune_and_fit_draw",
    "shard_count",
    "decode_shard_index",
    "init_shard",
    "run_fit_shard",
]


def shard_count() -> int:
    """Fit-array shards: one per (encoder, split, draw)."""
    return len(ENCODERS) * N_PATIENT_SPLITS * len(MAIN_DRAWS)


def decode_shard_index(shard_index: int) -> tuple[str, int, int]:
    """Decode a shard index into (encoder, split_index, draw_index)."""
    if shard_index not in range(shard_count()):
        raise ValueError(f"shard_index must be in [0, {shard_count() - 1}]")
    per_encoder = N_PATIENT_SPLITS * len(MAIN_DRAWS)
    enc_idx, rem = divmod(shard_index, per_encoder)
    split_idx, draw_pos = divmod(rem, len(MAIN_DRAWS))
    return ENCODERS[enc_idx], split_idx, MAIN_DRAWS[draw_pos]


def init_shard(
    config: dict[str, Any], split_idx: int, encoder: str
) -> tuple[pd.DataFrame, list[str], EvalPartition, dict[str, Path]]:
    """Load this split's per-encoder manifest, class names, and validation/test partitions."""
    paths = split_paths(ensure_dirs(config), split_idx)
    _, classes = load_train_identity(config, split_idx)
    m_file = paths["data"] / f"manifest_{encoder}.csv"
    train_df = pd.read_csv(m_file).query("split == 'train'").reset_index(drop=True)
    val_data = load_eval_partition(m_file, classes, "validation")
    test_data = load_eval_partition(m_file, classes, "test")
    evals = EvalPartition(*val_data, *test_data)
    return train_df, classes, evals, paths


def _pending_arms(paths: dict[str, Path], encoder: str, draw_idx: int) -> list[str]:
    return [
        a
        for a in ARMS
        if not (
            allocation_dir(paths, f"{encoder}/{a}", draw_idx) / RUN_RECORD_NAME
        ).exists()
    ]


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending arm of one (encoder, split, draw) shard."""
    encoder, split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx, encoder)
    pending = _pending_arms(paths, encoder, draw_idx)
    if not pending:
        return
    shard = _shard_context(
        train_df, names, split_idx, draw_idx, patients_per_class(config)
    )
    for arm in pending:
        fit_arm(
            config,
            allocation_dir(paths, f"{encoder}/{arm}", draw_idx),
            arm,
            shard,
            evals,
            draw_idx,
            FIT_SOURCE[arm],
        )
