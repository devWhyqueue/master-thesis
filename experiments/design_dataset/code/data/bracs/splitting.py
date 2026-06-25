"""Patient-disjoint BRACS split helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from data.bracs.metadata import LABELS

SPLITS = ("train", "validation", "test")


def write_seed_manifests(frame: pd.DataFrame, root: Path, seeds: list[int]) -> None:
    """Write patient-disjoint train/validation/test manifests for each seed."""
    root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        tagged = frame.merge(split_cases(frame, seed), on="case_id", how="inner")
        seed_dir = root / f"native_seed={seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            tagged[tagged["split"] == split].to_csv(
                seed_dir / f"{split}.csv", index=False
            )
        tagged.to_csv(seed_dir / "manifest_splits.csv", index=False)
        _write_json(seed_dir / "class_order.json", list(LABELS))
        counts = _train_counts(tagged)
        _write_json(seed_dir / "target_counts.json", counts)
        _write_json(seed_dir / "available_counts.json", counts)
        _write_json(seed_dir / "args.json", _args_payload(seed))
        assert_patient_disjoint(tagged)


def split_cases(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return approximate stratified patient-level split assignments."""
    case_labels = (
        frame.drop_duplicates(["case_id", "slide_id", "cancer_type"])
        .groupby("case_id")["cancer_type"]
        .agg(lambda values: Counter(values).most_common(1)[0][0])
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    rows = []
    for _, group in case_labels.groupby("cancer_type", sort=False):
        rows.extend(_split_group(group, rng))
    return pd.DataFrame(rows)


def assert_patient_disjoint(frame: pd.DataFrame) -> None:
    """Raise if any patient appears in multiple split labels."""
    split_counts = cast(pd.Series, frame.groupby("case_id")["split"].nunique())
    leaking = [
        str(case_id) for case_id, count in split_counts.items() if int(count) > 1
    ]
    if leaking:
        raise ValueError(f"BRACS patient leakage: {leaking[:5]}")


def _split_group(group: pd.DataFrame, rng: np.random.Generator) -> list[dict[str, str]]:
    cases = group["case_id"].astype(str).to_numpy()
    rng.shuffle(cases)
    n_cases = len(cases)
    n_train = (
        max(1, int(round(n_cases * 0.70))) if n_cases >= 3 else max(1, n_cases - 1)
    )
    n_val = max(1, int(round(n_cases * 0.15))) if n_cases >= 3 else 0
    if n_train + n_val >= n_cases and n_cases > 1:
        n_val = max(0, n_cases - n_train - 1)
    return (
        [{"case_id": case, "split": "train"} for case in cases[:n_train]]
        + [
            {"case_id": case, "split": "validation"}
            for case in cases[n_train : n_train + n_val]
        ]
        + [{"case_id": case, "split": "test"} for case in cases[n_train + n_val :]]
    )


def _train_counts(tagged: pd.DataFrame) -> dict[str, int]:
    counts = (
        tagged[tagged["split"] == "train"]
        .groupby("cancer_type")["slide_id"]
        .nunique()
        .reindex(LABELS, fill_value=0)
        .astype(int)
    )
    return {str(key): int(value) for key, value in counts.items()}


def _args_payload(seed: int) -> dict[str, object]:
    return {
        "dataset": "bracs",
        "seed": seed,
        "split_unit": "case_id",
        "target": "native_7_subtype",
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
