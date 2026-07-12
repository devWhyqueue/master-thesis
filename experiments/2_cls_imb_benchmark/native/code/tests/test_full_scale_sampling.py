import json

import pandas as pd

from data.full_scale.sampling import (
    assert_case_disjoint,
    available_training_counts,
    cap_patches,
    construct_training_split,
    max_feasible_total,
    power_law_counts,
    write_constructed_outputs,
)


def test_power_law_counts_keep_all_classes_and_total_size() -> None:
    available = pd.Series({"a": 5, "b": 4, "c": 3, "d": 2})
    counts = power_law_counts(available, ["a", "b", "c", "d"], 1.3, 10)

    assert sum(counts.values()) == 10
    assert set(counts) == {"a", "b", "c", "d"}
    assert min(counts.values()) >= 1
    assert all(counts[name] <= available[name] for name in counts)


def test_construct_training_split_preserves_size_and_classes() -> None:
    rows = [
        {"cancer_type": class_name, "slide_id": f"{class_name}_{index}", "patch_ids": []}
        for class_name, count in {"a": 5, "b": 4, "c": 3, "d": 2}.items()
        for index in range(count)
    ]
    frame = pd.DataFrame(rows)

    sampled = construct_training_split(frame, ["a", "b", "c", "d"], 1.0, 0, 12)

    assert sampled["slide_id"].nunique() == 12
    assert set(sampled["cancer_type"]) == {"a", "b", "c", "d"}
    assert sampled["slide_id"].is_unique


def test_power_law_counts_raise_on_infeasible_request() -> None:
    available = pd.Series({"a": 5, "b": 4, "c": 3, "d": 2})

    try:
        power_law_counts(available, ["a", "b", "c", "d"], 1.3, 15)
    except ValueError as error:
        assert "Infeasible power-law target" in str(error)
        assert "class=" in str(error)
    else:
        raise AssertionError("Expected infeasible target counts to raise ValueError.")


def test_max_feasible_total_returns_strict_upper_bound() -> None:
    available = pd.Series({"a": 5, "b": 4, "c": 3, "d": 2})

    total = max_feasible_total(available, ["a", "b", "c", "d"], 1.3)

    assert total == 12
    counts = power_law_counts(available, ["a", "b", "c", "d"], 1.3, total)

    assert sum(counts.values()) == total


def test_cap_patches_is_deterministic() -> None:
    frame = pd.DataFrame(
        [
            {
                "cancer_type": "a",
                "slide_id": "slide",
                "patch_ids": ["1_2", "0_9", "0_0", "1_0"],
            }
        ]
    )

    capped = cap_patches(frame, 3)

    assert capped.iloc[0]["patch_ids"] == ["0_0", "0_9", "1_0"]


def test_write_constructed_outputs_includes_combined_manifest(tmp_path) -> None:
    frame = pd.DataFrame(
        [{"cancer_type": "a", "slide_id": "slide", "patch_ids": ["0_0"]}]
    )

    write_constructed_outputs(
        {"train": frame, "validation": frame, "test": frame},
        {"a": 1},
        ["a"],
        str(tmp_path),
        {"seed": 0},
    )

    combined = pd.read_csv(tmp_path / "manifest_splits.csv")
    assert combined["split"].tolist() == ["train", "validation", "test"]


def test_available_training_counts_uses_native_full_pool(tmp_path) -> None:
    train = pd.DataFrame(
        [
            {"cancer_type": class_name, "slide_id": f"{class_name}_{index}"}
            for class_name, count in {"a": 3, "b": 1}.items()
            for index in range(count)
        ]
    )
    available = available_training_counts(train)
    assert available == {"a": 3, "b": 1}

    write_constructed_outputs(
        {"train": train.assign(patch_ids=[[] for _ in range(len(train))])},
        {"a": 2, "b": 1},
        ["a", "b"],
        str(tmp_path),
        {"seed": 0},
        available_counts=available,
    )
    with open(tmp_path / "available_counts.json") as file:
        assert json.load(file) == {"a": 3, "b": 1}


def test_case_disjoint_split_validation_accepts_patient_disjoint_splits() -> None:
    frame_by_split = {
        "train": pd.DataFrame({"case_id": ["case-a"], "slide_id": ["slide-a"]}),
        "validation": pd.DataFrame({"case_id": ["case-b"], "slide_id": ["slide-b"]}),
        "test": pd.DataFrame({"case_id": ["case-c"], "slide_id": ["slide-c"]}),
    }

    assert_case_disjoint(frame_by_split)


def test_case_disjoint_split_validation_rejects_patient_leakage() -> None:
    frame_by_split = {
        "train": pd.DataFrame({"case_id": ["case-a"], "slide_id": ["slide-a"]}),
        "validation": pd.DataFrame({"case_id": ["case-a"], "slide_id": ["slide-b"]}),
        "test": pd.DataFrame({"case_id": ["case-c"], "slide_id": ["slide-c"]}),
    }

    try:
        assert_case_disjoint(frame_by_split)
    except ValueError as error:
        assert "Case leakage between train and validation" in str(error)
    else:
        raise AssertionError("Expected case leakage to raise ValueError.")
