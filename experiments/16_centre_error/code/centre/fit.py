"""Fit stage: every arm of one (split, draw) shard, tuned on validation and evaluated on test."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.common import RUN_RECORD_NAME, write_run_record

from breadth import TIE_TOLERANCE
from breadth.fit import EvalPartition, _build_draw_record, init_shard, tune_and_fit_draw
from breadth.sampling import derive_draw_seed

from sites import allocation_dir

from centre import (
    ARMS,
    COHORT_SEED,
    KAPPA_FACTORS,
    N_DRAWS,
    N_SPLITS,
    PATIENT_COUNTS,
    patches_per_patient,
)
from centre.arms import (
    centre_error_parts,
    class_means,
    move_centres,
    noise_shift,
    whiten,
)
from centre.cohort import TrainingTable, draw_cohorts, training_table
from centre.pool import Pool, load_pool

__all__ = [
    "DrawInputs",
    "shard_count",
    "decode_shard_index",
    "split_arm",
    "arm_target",
    "run_fit_shard",
]

logger = logging.getLogger(__name__)

_NOISE_SEED_OFFSET = 1


class DrawInputs(NamedTuple):
    """Everything one draw's arms are built from."""

    tables: dict[int, TrainingTable]
    pool: Pool
    swap_centres: np.ndarray
    split_idx: int
    draw_idx: int


def shard_count() -> int:
    """Fit-array shards: one per (split, draw)."""
    return N_SPLITS * N_DRAWS


def decode_shard_index(shard_index: int) -> tuple[int, int]:
    """Decode a shard index into (split_index, draw_index)."""
    if shard_index not in range(shard_count()):
        raise ValueError(f"shard_index must be in [0, {shard_count() - 1}]")
    return divmod(shard_index, N_DRAWS)


def split_arm(arm: str) -> tuple[str, int]:
    """Arm name to (family, patient count), e.g. ``CW10`` -> (``CW``, 10)."""
    family = arm.rstrip("0123456789")
    return family, int(arm[len(family) :])


def _noise(inputs: DrawInputs, g: int) -> np.ndarray:
    """(C, d) class-centre errors of the size ``g`` patients produce, seeded per (split, draw, g, class)."""
    pool = inputs.pool
    shifts = []
    for c in range(len(pool.centres)):
        seed = derive_draw_seed(
            COHORT_SEED + _NOISE_SEED_OFFSET,
            inputs.split_idx,
            g,
            patches_per_patient(g),
            inputs.draw_idx,
            c,
        )
        dev = pool.deviations[pool.deviation_class == c]
        shifts.append(noise_shift(dev, g, np.random.default_rng(seed)))
    return np.stack(shifts)


def arm_target(arm: str, inputs: DrawInputs) -> np.ndarray | None:
    """(C, d) class centres an arm moves its cohort to; None for the real cohort."""
    family, g = split_arm(arm)
    pool, table = inputs.pool, inputs.tables[g]
    if family == "R":
        return None
    if family in ("C", "CW"):
        return pool.centres
    if family == "N":
        return pool.centres + _noise(inputs, g)
    if family == "Swap":
        return inputs.swap_centres
    means = class_means(table.x, table.y, len(pool.centres))
    parts = centre_error_parts(pool.centres - means, pool.discriminant)
    part = {
        "Glob": parts.shared,
        "Disc": parts.discriminant,
        "Off": parts.off_discriminant,
    }[family]
    return means + part


def _tune_whitened(
    x: np.ndarray, y: np.ndarray, evals: EvalPartition, pool: Pool
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Tune (kappa, lambda) on validation; ties go to the larger kappa, as for lambda."""
    best: tuple[Any, ...] | None = None
    best_score, best_kappa, scores = -1.0, 0.0, {}
    for factor in KAPPA_FACTORS:
        kappa = factor / float(pool.b_eigvals.mean())
        w_evals = replace(
            evals,
            val_x=whiten(evals.val_x, pool.b_basis, pool.b_eigvals, kappa),
            test_x=whiten(evals.test_x, pool.b_basis, pool.b_eigvals, kappa),
        )
        out = tune_and_fit_draw(
            whiten(x, pool.b_basis, pool.b_eigvals, kappa), y, w_evals
        )
        score = float(out[4]["patient_macro_balanced_accuracy"])
        scores[str(factor)] = score
        if best is None or score >= best_score - TIE_TOLERANCE:
            best, best_score, best_kappa = out, score, factor
    assert best is not None
    return best, {"kappa_factor": best_kappa, "kappa_validation_scores": scores}


def _fit_arm(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    inputs: DrawInputs,
    evals: EvalPartition,
) -> None:
    family, g = split_arm(arm)
    table = inputs.tables[g]
    target = arm_target(arm, inputs)
    x = table.x if target is None else move_centres(table.x, table.y, target)
    if family == "CW":
        out, extra = _tune_whitened(x, table.y, evals, inputs.pool)
    else:
        out, extra = tune_and_fit_draw(x, table.y, evals), {}
    fit, lam, test_preds, test_probs, val_end, test_end = out
    meta = (g, patches_per_patient(g), inputs.draw_idx)
    rec = _build_draw_record(
        config,
        meta,
        lam,
        fit,
        (test_preds, test_probs, val_end, test_end),
        evals.test_y,
    )
    write_run_record(out_dir, {**rec, "arm": arm, **extra}, keep_arrays=True)


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
    swap = training_table(train_df, names, cohorts.swap, 5)
    inputs = DrawInputs(
        {g: training_table(train_df, names, cohorts.nested, g) for g in PATIENT_COUNTS},
        load_pool(config, split_idx, names),
        class_means(swap.x, swap.y, len(names)),
        split_idx,
        draw_idx,
    )
    for arm in pending:
        logger.info("Fitting split %d, draw %d, arm %s", split_idx, draw_idx, arm)
        _fit_arm(config, allocation_dir(paths, arm, draw_idx), arm, inputs, evals)
