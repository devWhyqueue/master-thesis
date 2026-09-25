"""Fit stage: exp-39's seven paired arms (B/R10/R100/P10/P100/S10/S100) of one
(encoder, split, draw) shard, reusing exp-25/27's allocator/weighting and exp-5's
tuning/calibration machinery under a frozen local lambda grid (``transfer.LAMBDAS``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
from imbalance_benchmark.common import (
    EVAL_ARRAYS_NAME,
    N_PATIENT_SPLITS,
    RUN_RECORD_NAME,
    ensure_dirs,
    output_root,
    read_run_record,
    sign_file,
    split_paths,
    verify_signed_file,
    write_json,
)

from breadth.calibrate import TEMPERATURE_NAME
from breadth.fit import EvalPartition
from breadth.sampling import load_eval_partition

from sites import allocation_dir

from prevalence import patients_per_class
from prevalence.fit import _shard_context

from transfer import ARMS, ENCODERS, LAMBDAS, MAIN_DRAWS, SMOKE_DRAW
from transfer import extract, features, manifest
from transfer.fit.lock import fit_lock as _fit_lock
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
    "run_pilot_fit",
    "audit_fits",
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


def _completed_arm(
    result_dir: Path, arm: str, draw_idx: int, lock: dict[str, Any] | None = None
) -> bool:
    if not (result_dir / RUN_RECORD_NAME).exists():
        return False
    try:
        record = read_run_record(result_dir, array_fields=())
        assert record is not None
        candidates = record["candidates"]
        expected = list(LAMBDAS)
        if (
            record["arm"] != arm
            or record["grid"]["draw"] != draw_idx
            or not record["solver"]["converged"]
            or [c["lambda"] for c in candidates] != expected
            or record["selected_lambda"] not in expected
            or (draw_idx in MAIN_DRAWS and record.get("fit_lock") != lock)
        ):
            raise ValueError("run record does not match the frozen arm/grid")
        with np.load(result_dir / CANDIDATES_NAME, allow_pickle=False) as data:
            if data["lambdas"].tolist() != expected:
                raise ValueError("candidate coefficients do not match the frozen grid")
        with np.load(result_dir / EVAL_ARRAYS_NAME, allow_pickle=False) as data:
            if not {"test_labels", "test_preds", "test_probabilities"} <= set(
                data.files
            ):
                raise ValueError("test evaluation arrays are incomplete")
        temperature = json.loads((result_dir / TEMPERATURE_NAME).read_text())
        if temperature["selected_lambda"] != record["selected_lambda"]:
            raise ValueError("temperature uses a different lambda")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Incomplete or incompatible fit record: {result_dir}"
        ) from exc
    return True


def _pending_arms(
    paths: dict[str, Path], encoder: str, draw_idx: int, lock: dict[str, Any]
) -> list[str]:
    return [
        arm
        for arm in ARMS
        if not _completed_arm(
            allocation_dir(paths, f"{encoder}/{arm}", draw_idx), arm, draw_idx, lock
        )
    ]


def _pilot_train_df(
    config: dict[str, Any], encoder: str, train_df: pd.DataFrame
) -> pd.DataFrame:
    """Overlay reserved-draw feature refs onto the unchanged split-0 training rows."""
    train_df = train_df.copy()
    if encoder == "virchow2":
        source = manifest._full_manifest(config, 0)
        mask = source["split"].eq("train")
        refs = manifest._virchow2_reference_frame(config, 0).loc[mask]
        keys = list(features.IDENTITY_COLS)
        if (
            not train_df[keys]
            .astype(str)
            .equals(refs[keys].astype(str).reset_index(drop=True))
        ):
            raise RuntimeError("Pilot Virchow2 references changed row order")
        for column in ("feature_path", "feature_index"):
            train_df[column] = refs[column].reset_index(drop=True)
        return train_df

    audit_path = output_root(config) / "data" / "pilot_feature_audit.json"
    verify_signed_file(audit_path)
    frame = features.pilot_requested_frame(config)
    audit, index = extract.audit_uni2h(
        config, frame=frame, feature_root=extract.pilot_feature_root(config)
    )
    recorded = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit != recorded or audit["unresolved_missing"] or audit["unresolved_corrupt"]:
        raise RuntimeError("Pilot UNI2-h cache no longer matches its audit")
    for row_idx, key in enumerate(
        train_df[list(features.IDENTITY_COLS)]
        .astype(str)
        .itertuples(index=False, name=None)
    ):
        if key in index:
            (
                train_df.at[row_idx, "feature_path"],
                train_df.at[row_idx, "feature_index"],
            ) = index[key]
    return train_df


def _fit_cell(
    config: dict[str, Any], encoder: str, split_idx: int, draw_idx: int
) -> None:
    train_df, names, evals, paths = init_shard(config, split_idx, encoder)
    lock = _fit_lock(config, encoder, split_idx)
    if draw_idx == SMOKE_DRAW:
        train_df = _pilot_train_df(config, encoder, train_df)
    pending = _pending_arms(paths, encoder, draw_idx, lock)
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
            lock,
        )


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending arm of one main (encoder, split, draw) shard."""
    _fit_cell(config, *decode_shard_index(shard_index))


def run_pilot_fit(config: dict[str, Any], encoder: str) -> None:
    """Fit reserved draw 10000 on split 0, outside the main analysis grid."""
    if encoder not in ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder}")
    _fit_cell(config, encoder, 0, SMOKE_DRAW)


def audit_fits(config: dict[str, Any]) -> Path:
    """Record exact selected-arm completeness; reject absent or invalid fit evidence."""
    missing: list[str] = []
    checked = 0
    for split_idx in range(N_PATIENT_SPLITS):
        paths = split_paths(ensure_dirs(config), split_idx)
        for encoder in ENCODERS:
            lock = None
            for draw_idx in MAIN_DRAWS:
                for arm in ARMS:
                    result_dir = allocation_dir(paths, f"{encoder}/{arm}", draw_idx)
                    if (result_dir / RUN_RECORD_NAME).exists() and lock is None:
                        lock = _fit_lock(config, encoder, split_idx)
                    if _completed_arm(result_dir, arm, draw_idx, lock):
                        checked += 1
                    else:
                        missing.append(str(result_dir))
    out_path = output_root(config) / "data" / "fit_audit.json"
    write_json(
        out_path,
        {"expected": checked + len(missing), "valid": checked, "missing": missing},
    )
    sign_file(out_path)
    if missing:
        raise RuntimeError(f"{len(missing)} selected arm records are missing")
    return out_path
