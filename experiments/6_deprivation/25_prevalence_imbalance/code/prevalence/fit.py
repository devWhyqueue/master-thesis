"""Fit stage: prevalence-imbalance arms (r1..r100, N) of one (split, draw) shard.

Draws G eligible patients per class (20; config ``prevalence.patients_per_class`` overrides) at depth DEPTH = 160, fixed across every arm of a draw.
Ratio arms ``r{rho}`` reallocate the shared budget T = G x BALANCED x num_classes across classes with
exp-02's exponential-profile allocator, class order permuted per draw; ``N`` keeps native class
shares. Patients take a nested round-robin patch prefix (``centre.cohort.patient_rows``), so a
class's smaller-arm rows are always a prefix of its larger-arm rows for the same patient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from imbalance_benchmark.common import RUN_RECORD_NAME, write_json, write_run_record
from imbalance_benchmark.construction import _adjust_alloc, allocate_counts
from imbalance_benchmark.manifest.statistics import achieved_rho

from breadth.calibrate import (
    TEMPERATURE_NAME,
    _calibration_record,
    _logits,
    _store_if_inexact,
)
from breadth.fit import _build_draw_record, init_shard, tune_and_fit_draw
from breadth.sampling import (
    derive_draw_seed,
    eligible_patients_by_class,
    load_features_for_df,
)

from centre.cohort import patient_rows
from centre.fit import decode_shard_index, shard_count

from sites import allocation_dir

from prevalence import ARMS, BALANCED, DEPTH, G, PREVALENCE_SEED, patients_per_class

__all__ = [
    "decode_shard_index",
    "shard_count",
    "class_permutation",
    "class_counts",
    "run_fit_shard",
]


def class_permutation(split_idx: int, draw_idx: int, num_classes: int) -> np.ndarray:
    """Deterministic class-to-rank permutation for one draw's ratio arms."""
    seed = derive_draw_seed(PREVALENCE_SEED, split_idx, num_classes, DEPTH, draw_idx, 0)
    return np.random.default_rng(seed).permutation(num_classes)


def _draw_patients(
    train_df: pd.DataFrame, names: list[str], split_idx: int, draw_idx: int, g: int
) -> list[list[str]]:
    """g eligible patients per class, fixed across every arm of this draw."""
    chosen = []
    for ci, name in enumerate(names):
        eligible = eligible_patients_by_class(train_df, name, DEPTH)
        seed = derive_draw_seed(PREVALENCE_SEED, split_idx, g, DEPTH, draw_idx, ci)
        rng = np.random.default_rng(seed)
        chosen.append([str(p) for p in rng.choice(eligible, size=g, replace=False)])
    return chosen


def class_counts(
    arm: str,
    perm: np.ndarray,
    available: list[int],
    pool_counts: list[int],
    g: int = G,
) -> list[int]:
    """Per-class total patch counts for one arm at the shared budget T = g x BALANCED x K."""
    num_classes = len(available)
    total = g * BALANCED * num_classes
    if arm == "N":
        share = sum(pool_counts)
        target = [total * c / share for c in pool_counts]
        allocated = [min(max(round(t), g), a) for t, a in zip(target, available)]
        _adjust_alloc(allocated, available, target, total - sum(allocated), g)
        return allocated
    ranked = allocate_counts(available, total, float(arm[1:]), g)
    counts = [0] * num_classes
    for rank, cls_idx in enumerate(perm):
        counts[cls_idx] = ranked[rank]
    return counts


def _patient_counts(total: int, g: int = G) -> list[int]:
    """Split one class's total across its g patients; nesting is per-patient (patient_rows), not here."""
    base, remainder = divmod(total, g)
    return [base + 1] * remainder + [base] * (g - remainder)


def _arm_rows(
    train_df: pd.DataFrame,
    names: list[str],
    patients: list[list[str]],
    counts: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Features and integer targets of one arm's realized allocation."""
    rows: list[int] = []
    y: list[int] = []
    for ci, name in enumerate(names):
        class_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == name])
        class_rows: list[int] = []
        for patient, m in zip(
            patients[ci], _patient_counts(counts[ci], len(patients[ci]))
        ):
            if m:
                class_rows.extend(patient_rows(class_df, [patient], m))
        rows.extend(class_rows)
        y.extend([ci] * len(class_rows))
    x = load_features_for_df(train_df.loc[rows]).astype(np.float64)
    return x, np.asarray(y, dtype=np.int64)


def _write_temperature(
    out_dir: Path,
    evals: Any,
    fit: Any,
    test_preds: np.ndarray,
    test_probs: np.ndarray,
    lam: float,
) -> None:
    """Fit a validation temperature from the in-memory fit and store it alongside the run record."""
    val_logits = _logits(evals.val_x, fit.coef, fit.intercept)
    test_logits = _logits(evals.test_x, fit.coef, fit.intercept)
    payload, scaled = _calibration_record(evals, val_logits, test_logits, test_preds)
    payload["max_abs_scaled_probability_error"] = _store_if_inexact(
        out_dir, scaled, test_probs, payload["temperature"]
    )
    write_json(out_dir / TEMPERATURE_NAME, {**payload, "selected_lambda": lam})


class _Shard(NamedTuple):
    """Everything one (split, draw)'s arms are fit from: fixed across every arm."""

    train_df: pd.DataFrame
    names: list[str]
    patients: list[list[str]]
    perm: np.ndarray
    available: list[int]
    pool_counts: list[int]
    g: int


def _shard_context(
    train_df: pd.DataFrame, names: list[str], split_idx: int, draw_idx: int, g: int
) -> _Shard:
    """Patient draws, class permutation, per-class availability cap, and native pool counts."""
    return _Shard(
        train_df,
        names,
        _draw_patients(train_df, names, split_idx, draw_idx, g),
        class_permutation(split_idx, draw_idx, len(names)),
        [g * DEPTH] * len(names),
        [int((train_df["cancer_type"] == name).sum()) for name in names],
        g,
    )


def _fit_arm(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    shard: _Shard,
    evals: Any,
    draw_idx: int,
) -> None:
    """Allocate one arm's patch counts, fit it, and write its run record and temperature."""
    counts = class_counts(arm, shard.perm, shard.available, shard.pool_counts, shard.g)
    x, y = _arm_rows(shard.train_df, shard.names, shard.patients, counts)
    fit, lam, test_preds, test_probs, val_end, test_end = tune_and_fit_draw(x, y, evals)
    rec = _build_draw_record(
        config,
        (shard.g, DEPTH, draw_idx),
        lam,
        fit,
        (test_preds, test_probs, val_end, test_end),
        evals.test_y,
    )
    named_counts = dict(zip(shard.names, (int(c) for c in counts)))
    extra = {
        "arm": arm,
        "class_counts": named_counts,
        "realized_rho": achieved_rho(named_counts),
    }
    write_run_record(out_dir, {**rec, **extra}, keep_arrays=True)
    _write_temperature(out_dir, evals, fit, test_preds, test_probs, lam)


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending arm of one (split, draw)."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    pending = [
        a
        for a in ARMS
        if not (allocation_dir(paths, a, draw_idx) / RUN_RECORD_NAME).exists()
    ]
    if not pending:
        return
    shard = _shard_context(
        train_df, names, split_idx, draw_idx, patients_per_class(config)
    )
    for arm in pending:
        _fit_arm(
            config, allocation_dir(paths, arm, draw_idx), arm, shard, evals, draw_idx
        )
