from __future__ import annotations

import json
import sqlite3
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.calibration import temperature_scaled_probabilities
from imbalance_benchmark.common import read_run_record

__all__ = [
    "load_runs_frame",
    "load_eval_details",
    "load_classwise",
    "load_split_payload",
    "load_test_identity",
    "load_seed_predictions",
    "EXPECTED_CONFIRMATION_SEEDS",
]

# Every locked patient split and tail assignment is repeated with exactly five
# disjoint confirmation initialization seeds (report §"Training, selection, and
# replication"). Inference must not proceed on a silently truncated seed block.
EXPECTED_CONFIRMATION_SEEDS = 5


def load_runs_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the ``runs`` table."""
    return pd.read_sql_query("SELECT * FROM runs", conn)


def load_eval_details(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load ``runs`` joined with ``eval_results``, one row per run x split."""
    details = pd.read_sql_query(
        "SELECT r.*, e.split, e.accuracy, e.balanced_accuracy, e.macro_precision, "
        "e.macro_recall, e.macro_f1, e.macro_nll, e.negative_log_likelihood, "
        "e.brier_score, e.expected_calibration_error, e.quadratic_weighted_kappa, "
        "e.ordinal_mean_absolute_error, e.extended_json "
        "FROM runs r JOIN eval_results e ON r.run_id = e.run_id",
        conn,
    )
    extended = details["extended_json"].map(lambda raw: json.loads(raw) if raw else {})
    endpoint_details = extended.map(_canonical_endpoint_details)
    return pd.concat(
        [
            details.drop(columns="extended_json"),
            pd.json_normalize(list(endpoint_details)),
        ],
        axis=1,
    )


def _canonical_endpoint_details(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten clustered and locked-tier endpoints for equal-split reporting."""
    out = dict(payload.get("clustered_endpoints", {}))
    # Same value as the split's own ECE column (both computed from the same
    # labels/probabilities); keeping it here would collide with that column
    # once the two frames are concatenated below.
    out.pop("expected_calibration_error", None)
    for tier, metrics in payload.get("tier_metrics", {}).items():
        out.update(
            {f"tier_{tier}_{metric}": value for metric, value in metrics.items()}
        )
    out.update(
        {
            key: payload[key]
            for key in (
                "temperature",
                "temperature_scaled_nll",
                "temperature_scaled_brier",
                "temperature_scaled_ece",
                "temperature_scaled_ece_ci",
            )
            if key in payload
        }
    )
    return out


def load_classwise(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load ``eval_classwise`` joined with ``runs`` identity columns."""
    return pd.read_sql_query(
        "SELECT r.run_id, r.benchmark, r.condition, r.assignment, r.method, r.seed_index, c.split, "
        "c.class_name, c.tier, c.precision, c.recall, c.f1, c.support, c.nll, c.brier "
        "FROM eval_classwise c JOIN runs r ON r.run_id = c.run_id",
        conn,
    )


def load_split_payload(result_dir: Path, split: str) -> dict[str, Any] | None:
    """Load one run's evaluated split (with labels/preds/probabilities/logits reattached)."""
    record = read_run_record(result_dir, splits=(split,))
    if record is None:
        return None
    return record.get("splits", {}).get(split)


@cache
def load_test_identity(
    manifest_path: Path, is_mil: bool, split_name: str = "test"
) -> pd.DataFrame:
    """Rebuild the case_id/slide_id vector for a locked split in prediction-array order.

    Confirmation's ``DataLoader`` never shuffles, so this reproduces the exact
    row order ``ImbalanceDataset``/``BagFeatureDataset`` iterate in (one row
    per patch, or one row per slide via ``groupby("slide_id").first()``),
    letting every run's stored ``preds``/``labels``/``probabilities`` arrays be
    joined back to patient/slide identity by position without persisting
    patient IDs inside every run record.
    """
    identity_columns = {"case_id", "slide_id", "cancer_type", "split"}
    df = pd.read_csv(manifest_path, usecols=lambda column: column in identity_columns)
    df["case_id"] = df["case_id"].astype(str)
    df = cast(pd.DataFrame, df[df["split"] == split_name]).reset_index(drop=True)
    if is_mil:
        df = df.groupby("slide_id", sort=False).first().reset_index()
    return cast(pd.DataFrame, df[["case_id", "slide_id", "cancer_type"]]).reset_index(
        drop=True
    )


def _array_fields_for(fields: tuple[str, ...]) -> tuple[str, ...]:
    """Narrow the heavy NPZ sidecar read to only what ``fields`` will build."""
    array_fields = {"labels"}
    if "preds" in fields:
        array_fields.add("preds")
    if "probs" in fields or "temperature_scaled_probs" in fields:
        array_fields.add("probabilities")
    if "temperature_scaled_probs" in fields:
        array_fields.add("logits")
    return tuple(sorted(array_fields))


def _require_complete_confirmation_block(
    method_dir: Path, seed_dirs: list[Path], array_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Refuse to stack a partial confirmation block; report missing/failed seeds.

    A method directory that exists must carry exactly the five confirmation
    seeds, each with a readable ``test`` split. Missing or unreadable seeds are
    an implementation failure the analysis must surface, not silently average
    over (report §"Training, selection, and replication").
    """
    present = {int(d.name.split("=")[1]) for d in seed_dirs}
    expected = set(range(EXPECTED_CONFIRMATION_SEEDS))
    records = [
        read_run_record(directory, splits=("test",), array_fields=array_fields)
        for directory in seed_dirs
    ]
    unreadable = {
        int(directory.name.split("=")[1])
        for directory, record in zip(seed_dirs, records, strict=True)
        if record is None or "test" not in record.get("splits", {})
    }
    missing = sorted((expected - present) | unreadable)
    extra = sorted(present - expected)
    if missing or extra:
        raise RuntimeError(
            f"Confirmation block at {method_dir} is incomplete: "
            f"expected seeds {sorted(expected)}, "
            f"missing/failed {missing}, unexpected {extra}. "
            "Inference requires exactly five valid confirmation runs."
        )
    return [record for record in records if record is not None]


def load_seed_predictions(
    paths: dict[str, Path],
    condition: str,
    method: str,
    assignment: str = "native",
    fields: tuple[str, ...] = ("preds", "probs"),
) -> dict[str, Any] | None:
    """Stack one method's confirmed test-split predictions across its confirmation seeds.

    ``fields`` selects which of ``"preds"``, ``"probs"``, and
    ``"temperature_scaled_probs"`` to load and stack; narrowing it skips the
    matching NPZ arrays entirely. The result is memoized per ``(method_dir,
    fields)`` -- callers must treat every returned array as read-only.
    """
    method_dir = _confirmation_dir(paths, condition, method, assignment)
    if not method_dir.exists():
        raise RuntimeError(
            f"Required confirmation method '{method}' is missing for "
            f"{assignment}/{condition}: {method_dir}"
        )
    return _load_confirmation_block(method_dir, fields)


@lru_cache(maxsize=4)
def _load_confirmation_block(
    method_dir: Path, fields: tuple[str, ...]
) -> dict[str, Any] | None:
    seed_dirs = sorted(
        method_dir.glob("seed=*"), key=lambda p: int(p.name.split("=")[1])
    )
    if not seed_dirs:
        raise RuntimeError(f"Confirmation block is missing seed runs: {method_dir}")
    records = _require_complete_confirmation_block(
        method_dir, seed_dirs, _array_fields_for(fields)
    )
    if not records:
        return None
    test_splits = [r["splits"]["test"] for r in records]
    arrays = _stack_prediction_arrays(test_splits, fields)
    return {
        "class_names": records[0].get("class_names", []),
        "labels": np.array(test_splits[0]["labels"]),
        **arrays,
    }


def _stack_prediction_arrays(
    test_splits: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Stack only the requested seed-indexed prediction arrays."""
    arrays: dict[str, np.ndarray] = {}
    if "preds" in fields:
        arrays["preds"] = np.stack([np.array(split["preds"]) for split in test_splits])
    if "probs" in fields:
        arrays["probs"] = np.stack(
            [np.array(split["probabilities"]) for split in test_splits]
        )
    if "temperature_scaled_probs" in fields:
        arrays["temperature_scaled_probs"] = np.stack(
            [temperature_scaled_probabilities(split) for split in test_splits]
        )
    return arrays


def _confirmation_dir(
    paths: dict[str, Path], condition: str, method: str, assignment: str
) -> Path:
    """Resolve one assignment-aware directory, sharing the unassigned baseline."""
    assigned = paths["results"] / f"assignment={assignment}" / condition / method
    unassigned = paths["results"] / "assignment=unassigned" / condition / method
    if assigned.exists():
        return assigned
    if unassigned.exists():
        return unassigned
    return paths["results"] / condition / method
