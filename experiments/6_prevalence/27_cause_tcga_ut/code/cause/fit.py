"""Fit stage: exp-27's new P/S arms of one (split, draw) shard, plus their LP/Lr logit-adjusted
derivatives. r1/r{rho} themselves are reused from exp-25's stored outputs, not fit here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    ensure_dirs,
    read_run_record,
    split_paths,
    write_run_record,
)

from breadth.fit import init_shard

from centre.fit import decode_shard_index, shard_count

from sites import allocation_dir

from spectrum import baseline_config

from prevalence import patients_per_class
from prevalence.fit import _fit_arm, _shard_context

from cause import ADJUST_SOURCE, ADJUSTED_ARMS, FIT_SOURCE, NEW_FIT_ARMS

__all__ = ["decode_shard_index", "shard_count", "run_fit_shard"]

_PROBABILITY_FLOOR = np.finfo(np.float64).tiny


def _adjusted_preds(
    probs: np.ndarray, prior_counts: dict[str, int], names: list[str]
) -> np.ndarray:
    """argmax(log p - log pi): remove each class's training-prior share from its log-probability."""
    shares = np.array([prior_counts[n] for n in names], dtype=np.float64)
    shares = shares / shares.sum()
    adjusted = np.log(np.maximum(probs, _PROBABILITY_FLOOR)) - np.log(shares)
    return np.argmax(adjusted, axis=1)


def _write_adjusted(
    own_paths: dict[str, Path],
    exp25_paths: dict[str, Path],
    arm: str,
    draw_idx: int,
    names: list[str],
) -> None:
    """Write one LP/Lr arm's run record from an already-fit arm's stored test probabilities."""
    out_dir = allocation_dir(own_paths, arm, draw_idx)
    if (out_dir / RUN_RECORD_NAME).exists():
        return
    source_arm, origin = ADJUST_SOURCE[arm]
    src_dir = allocation_dir(
        own_paths if origin == "own" else exp25_paths, source_arm, draw_idx
    )
    rec = read_run_record(
        src_dir, splits=("test",), array_fields=("labels", "probabilities")
    )
    if rec is None:
        raise RuntimeError(f"Missing source run record at {src_dir}")
    # Pρ stores the prior it was reweighted toward; rρ's own realized class shares stand in
    # for its (unweighted) implicit training prior.
    prior_counts = rec["prior_counts"] if origin == "own" else rec["class_counts"]
    test = rec["splits"]["test"]
    labels = np.asarray(test["labels"])
    probs = np.asarray(test["probabilities"])
    preds = _adjusted_preds(probs, prior_counts, names)
    write_run_record(
        out_dir,
        {
            "arm": arm,
            "adjusted_from": source_arm,
            "prior_counts": prior_counts,
            "splits": {
                "test": {"labels": labels, "preds": preds, "probabilities": probs}
            },
        },
        keep_arrays=True,
    )


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending P/S arm of one (split, draw), then write its pending LP/Lr arms."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    pending = [
        a
        for a in NEW_FIT_ARMS
        if not (allocation_dir(paths, a, draw_idx) / RUN_RECORD_NAME).exists()
    ]
    if pending:
        shard = _shard_context(
            train_df, names, split_idx, draw_idx, patients_per_class(config)
        )
        for arm in pending:
            data_arm, prior_arm = FIT_SOURCE[arm]
            _fit_arm(
                config,
                allocation_dir(paths, arm, draw_idx),
                arm,
                shard,
                evals,
                draw_idx,
                data_arm=data_arm,
                prior_arm=prior_arm,
            )
    exp25_paths = split_paths(
        ensure_dirs(baseline_config(config, "prevalence_outputs")), split_idx
    )
    for arm in ADJUSTED_ARMS:
        _write_adjusted(paths, exp25_paths, arm, draw_idx, names)
