"""Fit stage: every arm of one (split, draw) shard, R/C/N/CW via centre, RW/RWc via directions."""

from __future__ import annotations

from typing import Any

import numpy as np
from imbalance_benchmark.common import RUN_RECORD_NAME

from breadth.fit import init_shard

from sites import allocation_dir

from centre import PATIENT_COUNTS
from centre.cohort import draw_cohorts, training_table
from centre.fit import (
    DrawInputs,
    _fit_arm as _fit_centre_arm,
    decode_shard_index,
    split_arm,
)
from centre.pool import load_pool

from directions.fit import _fit_arm as _fit_directions_arm

from mechanism import ARMS

__all__ = ["run_fit_shard"]

_DIRECTIONS_FAMILIES = ("RW", "RWc")


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every arm of one (split, draw) whose run record does not exist yet."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    pending = [
        a
        for a in ARMS
        if not (allocation_dir(paths, a, draw_idx) / RUN_RECORD_NAME).exists()
    ]
    if not pending:
        return
    cohorts = draw_cohorts(train_df, names, split_idx, draw_idx)
    tables = {
        g: training_table(train_df, names, cohorts.nested, g) for g in PATIENT_COUNTS
    }
    pool = load_pool(config, split_idx, names)
    inputs = DrawInputs(tables, pool, np.empty(0), split_idx, draw_idx)
    for arm in pending:
        family, g = split_arm(arm)
        out_dir = allocation_dir(paths, arm, draw_idx)
        if family in _DIRECTIONS_FAMILIES:
            _fit_directions_arm(
                config, out_dir, arm, tables[g], len(names), pool, evals, draw_idx
            )
        else:
            _fit_centre_arm(config, out_dir, arm, inputs, evals)
