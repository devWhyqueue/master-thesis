"""Fit stage: every arm of one (setting, split, draw) shard, pilot or main phase.

Reuses exp-25/27/28's allocator, patient draws, and prior/support reweighting
(``prevalence.fit``) unchanged; only the training/validation/test features are shifted by the
frozen ``x^(alpha)`` intervention (``separation.geometry``) before ``tune_and_fit_draw``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from imbalance_benchmark.common import RUN_RECORD_NAME, write_run_record

from breadth.fit import _build_draw_record, init_shard, tune_and_fit_draw

from sites import allocation_dir

from prevalence import DEPTH
from prevalence.fit import (
    _Shard,
    _arm_rows,
    _draw_patients,
    _prior_weights,
    _shard_context,
    class_counts,
    class_permutation,
)

from separation import (
    ARMS,
    FIT_SOURCE,
    G,
    MAIN_DRAWS,
    N_SPLITS,
    PILOT_DRAWS,
    main_settings,
    pilot_settings,
)
from separation.geometry import apply_intervention, load_centres
from separation.precheck import load_alpha

__all__ = [
    "pilot_shard_count",
    "main_shard_count",
    "decode_pilot_shard_index",
    "decode_main_shard_index",
    "run_fit_shard",
]

_ALPHA_KEY: dict[str, str | None] = {
    "expanded_bracs": "alpha_expand",
    "tcga_native_10": None,
    "tcga_contracted_10": "alpha_contract",
    "native_bracs_replay": None,
}


def pilot_shard_count(dataset: str) -> int:
    """Pilot-array shards for this dataset: one per (split, setting), draw fixed at 0."""
    return N_SPLITS * len(pilot_settings(dataset))


def main_shard_count(dataset: str) -> int:
    """Main-array shards for this dataset: one per (split, setting, draw)."""
    return N_SPLITS * len(main_settings(dataset)) * len(MAIN_DRAWS)


def decode_pilot_shard_index(dataset: str, shard_index: int) -> tuple[str, int, int]:
    """Decode a pilot shard index into (setting, split_idx, draw_idx = 0)."""
    settings = pilot_settings(dataset)
    count = N_SPLITS * len(settings)
    if shard_index not in range(count):
        raise ValueError(f"shard_index must be in [0, {count - 1}]")
    split_idx, setting_idx = divmod(shard_index, len(settings))
    return settings[setting_idx], split_idx, PILOT_DRAWS[0]


def decode_main_shard_index(dataset: str, shard_index: int) -> tuple[str, int, int]:
    """Decode a main shard index into (setting, split_idx, draw_idx)."""
    settings = main_settings(dataset)
    per_split = len(settings) * len(MAIN_DRAWS)
    count = N_SPLITS * per_split
    if shard_index not in range(count):
        raise ValueError(f"shard_index must be in [0, {count - 1}]")
    split_idx, rem = divmod(shard_index, per_split)
    setting_idx, draw_pos = divmod(rem, len(MAIN_DRAWS))
    return settings[setting_idx], split_idx, MAIN_DRAWS[draw_pos]


def _tcga10_shard(train_df, names, split_idx: int, draw_idx: int) -> _Shard:
    """TCGA-UT's own 10-patient nested prefix of exp-25/27's paired 20-patient draw."""
    patients20 = _draw_patients(train_df, names, split_idx, draw_idx, 20)
    patients10 = [p[:G] for p in patients20]
    perm = class_permutation(split_idx, draw_idx, len(names))
    pool_counts = [int((train_df["cancer_type"] == name).sum()) for name in names]
    return _Shard(
        train_df, names, patients10, perm, [G * DEPTH] * len(names), pool_counts, G
    )


def _bracs_shard(train_df, names, split_idx: int, draw_idx: int) -> _Shard:
    """BRACS's own G = 10 patient draw (exp-26/28's own seed, shared by every BRACS setting)."""
    return _shard_context(train_df, names, split_idx, draw_idx, G)


def _setting_shard(
    setting: str, train_df, names, split_idx: int, draw_idx: int
) -> _Shard:
    if setting in ("expanded_bracs", "native_bracs_replay"):
        return _bracs_shard(train_df, names, split_idx, draw_idx)
    return _tcga10_shard(train_df, names, split_idx, draw_idx)


def _alpha_for_split(precheck: dict[str, Any], setting: str, split_idx: int) -> float:
    key = _ALPHA_KEY[setting]
    return 1.0 if key is None else float(precheck[key][split_idx])


def _fit_arm(
    config: dict[str, Any],
    out_dir,
    arm: str,
    shard: _Shard,
    evals: Any,
    centres: Any,
    alpha: float,
    draw_idx: int,
    setting: str,
) -> None:
    data_arm, prior_arm = FIT_SOURCE[arm]
    counts = class_counts(
        data_arm, shard.perm, shard.available, shard.pool_counts, shard.g
    )
    x, y = _arm_rows(shard.train_df, shard.names, shard.patients, counts)
    x = apply_intervention(x, y, centres, alpha)
    w_evals = evals
    if alpha != 1.0:
        w_evals = replace(
            evals,
            val_x=apply_intervention(evals.val_x, evals.val_y, centres, alpha),
            test_x=apply_intervention(evals.test_x, evals.test_y, centres, alpha),
        )
    prior_counts = None
    weight = None
    if prior_arm is not None:
        prior_counts = class_counts(
            prior_arm, shard.perm, shard.available, shard.pool_counts, shard.g
        )
        weight = _prior_weights(counts, prior_counts)
    fit, lam, test_preds, test_probs, val_end, test_end = tune_and_fit_draw(
        x, y, w_evals, weight
    )
    rec = _build_draw_record(
        config,
        (shard.g, DEPTH, draw_idx),
        lam,
        fit,
        (test_preds, test_probs, val_end, test_end),
        w_evals.test_y,
    )
    extra: dict[str, Any] = {
        "arm": arm,
        "setting": setting,
        "alpha": alpha,
        "class_counts": dict(zip(shard.names, (int(c) for c in counts))),
    }
    if prior_counts is not None:
        extra["prior_counts"] = dict(zip(shard.names, (int(c) for c in prior_counts)))
    write_run_record(out_dir, {**rec, **extra}, keep_arrays=True)


def _decode_shard(dataset: str, phase: str, shard_index: int) -> tuple[str, int, int]:
    if phase == "pilot":
        return decode_pilot_shard_index(dataset, shard_index)
    return decode_main_shard_index(dataset, shard_index)


def _pending_arms(paths: dict[str, Any], setting: str, draw_idx: int) -> list[str]:
    return [
        arm
        for arm in ARMS
        if not (
            allocation_dir(paths, f"{setting}_{arm}", draw_idx) / RUN_RECORD_NAME
        ).exists()
    ]


def run_fit_shard(config: dict[str, Any], phase: str, shard_index: int) -> None:
    """Fit every pending arm of one (setting, split, draw) shard."""
    dataset = config["dataset"]["name"]
    setting, split_idx, draw_idx = _decode_shard(dataset, phase, shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    pending = _pending_arms(paths, setting, draw_idx)
    if not pending:
        return
    shard = _setting_shard(setting, train_df, names, split_idx, draw_idx)
    centres = load_centres(config, split_idx, names)
    alpha = _alpha_for_split(load_alpha(config), setting, split_idx)
    for arm in pending:
        _fit_arm(
            config,
            allocation_dir(paths, f"{setting}_{arm}", draw_idx),
            arm,
            shard,
            evals,
            centres,
            alpha,
            draw_idx,
            setting,
        )
