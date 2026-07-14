from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imbalance_benchmark.analysis.inference.crossed_permutation import (
    crossed_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.inference.preflight import run_preflight
from imbalance_benchmark.construction import allocate_counts, effective_rho, max_shared_total
from imbalance_benchmark.manifest.freezing import _build_conditions
from imbalance_benchmark.manifest.pilot import meets_method_floor


def _patches(class_name: str, n_patients: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [f"{class_name}_patient_{patient}" for patient in range(n_patients) for _ in range(10)],
            "slide_id": [
                f"{class_name}_patient_{patient}_slide_{patch % 2}"
                for patient in range(n_patients)
                for patch in range(10)
            ],
            "patch_id": [f"{class_name}_{patient}_{patch}" for patient in range(n_patients) for patch in range(10)],
            "cancer_type": class_name,
            "split": "train",
        }
    )


def test_asymmetric_availability_keeps_the_largest_shared_total() -> None:
    available = [1000, 500, 200]

    total = max_shared_total(available, min_support=20)
    rho = effective_rho(available, rho=100.0, min_support=20, total_t=total)
    allocation = allocate_counts(available, total, rho, min_support=20)

    assert total == 600
    assert 1.0 < rho < 100.0
    assert sum(allocation) == total
    assert min(allocation) >= 20
    assert all(count <= capacity for count, capacity in zip(allocation, available, strict=True))


def test_evidence_seed_is_stable_when_a_semantic_class_changes_tail_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = pd.concat([_patches("A"), _patches("B")], ignore_index=True)
    observed: list[tuple[str, int]] = []

    def selector(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        observed.append((str(df["cancer_type"].iloc[0]), seed))
        return df.iloc[:n]

    monkeypatch.setattr("imbalance_benchmark.manifest.freezing.select_patches_round_robin", selector)
    _build_conditions(frame, ["A", "B"], 100, 20, False, 17, tmp_path, condition_names=("moderate",))
    first = dict(observed)
    observed.clear()
    _build_conditions(frame, ["B", "A"], 100, 20, False, 17, tmp_path, condition_names=("moderate",))

    assert dict(observed) == first


def test_patch_conditions_record_one_fixed_patient_slide_pool(tmp_path: Path) -> None:
    frame = pd.concat([_patches("A"), _patches("B")], ignore_index=True)

    conditions = _build_conditions(frame, ["A", "B"], 200, 20, False, 8, tmp_path)

    assert conditions["balanced"]["evidence_pool_hash"] == conditions["moderate"]["evidence_pool_hash"]
    assert conditions["moderate"]["evidence_pool_hash"] == conditions["severe"]["evidence_pool_hash"]


def test_method_floor_requires_patients_and_slides_together() -> None:
    assert not meets_method_floor({"patients": 9, "slides": 100}, patient_equals_slide=False)
    assert not meets_method_floor({"patients": 100, "slides": 19}, patient_equals_slide=False)
    assert meets_method_floor({"patients": 10, "slides": 20}, patient_equals_slide=False)


def test_preflight_is_descriptive_when_any_split_class_fails_kish_threshold() -> None:
    rows = []
    for split, n_patients in ((0, 2), (1, 10)):
        for class_name in ("A", "B"):
            rows.extend(
                {
                    "case_id": f"{split}_{class_name}_{patient}",
                    "cancer_type": class_name,
                    "patient_split": split,
                }
                for patient in range(n_patients)
            )

    result = run_preflight(pd.DataFrame(rows), n_replicates=40, seed=4)

    assert result["by_split_class"]["0"]["A"]["kish_effective_count"] < 5
    assert result["is_descriptive_only"]


def test_crossed_tail_permutation_accepts_a_locked_tail_for_each_split() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels]
    methods = np.stack([probabilities, probabilities])
    ce = np.stack([probabilities[[1, 2, 0, 1, 2, 0]], probabilities])
    blocks = [
        (labels, methods, ce, np.array([f"a{index}" for index in range(6)])),
        (labels, methods, ce, np.array([f"b{index}" for index in range(6)])),
    ]

    p_value = crossed_block_permutation_tail_nll(blocks, [[2], [1]], n_permutations=32, seed=3)

    assert 0.0 <= p_value <= 1.0
