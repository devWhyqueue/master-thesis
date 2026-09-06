from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diversity import manifests
from diversity.manifests import allocation as manifests_allocation

CLASS_NAMES = ["A", "B"]
CASES_PER_CLASS = 2
POOL_SIZE = 6
PINNED_N = 2


def _write_feature(path: Path, value: float, dim: int = 4) -> str:
    vector = torch.tensor([[value, *([0.0] * (dim - 1))]], dtype=torch.float32)
    torch.save(vector, path)
    return str(path)


def _synthetic_manifest(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A tiny frozen train split plus one already-selected 'balanced' condition.

    Each (class, case, slide) slot has a pool of POOL_SIZE patches spread
    along one feature dimension at values 0..POOL_SIZE-1, so narrow (nearest
    the slot mean) and wide (farthest-point) selections are hand-checkable
    and, with headroom POOL_SIZE/PINNED_N > 1, must differ.
    """
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    rows = []
    for cls in CLASS_NAMES:
        for case in range(CASES_PER_CLASS):
            case_id = f"{cls}_case{case}"
            slide_id = f"{case_id}_slide0"
            for patch in range(POOL_SIZE):
                patch_id = f"{case_id}_p{patch}"
                path = feature_dir / f"{patch_id}.pt"
                rows.append(
                    {
                        "cancer_type": cls,
                        "case_id": case_id,
                        "slide_id": slide_id,
                        "patch_id": patch_id,
                        "feature_path": _write_feature(path, float(patch)),
                        "feature_index": 0,
                        "split": "train",
                    }
                )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(tmp_path / "manifest.csv", index=False)

    # The 'already selected' condition: the first PINNED_N patches per slot,
    # grouped by class in CLASS_NAMES order (mirrors write_condition).
    source_rows = []
    for cls in CLASS_NAMES:
        cls_rows = []
        for case in range(CASES_PER_CLASS):
            case_id = f"{cls}_case{case}"
            slot = manifest[
                (manifest["cancer_type"] == cls) & (manifest["case_id"] == case_id)
            ].iloc[:PINNED_N]
            cls_rows.append(slot)
        source_rows.append(pd.concat(cls_rows, ignore_index=True))
    source_df = pd.concat(source_rows, ignore_index=True)
    source_df.to_csv(tmp_path / "manifest_balanced.csv", index=False)
    return manifest, source_df


def _exp2_freeze(source_df: pd.DataFrame) -> dict:
    allocated_counts = {
        cls: int((source_df["cancer_type"] == cls).sum()) for cls in CLASS_NAMES
    }
    return {
        "min_support": PINNED_N,
        "conditions": {
            "balanced": {
                "allocated_counts": allocated_counts,
                "evidence_pool_hash": "test-pool-hash",
                "construction_seed": 0,
                "spread_classes": None,
                "spread_ratio": None,
                "spread_tail_classes": None,
            }
        },
    }


def test_select_narrow_picks_nearest_to_mean_with_feature_index_tiebreak() -> None:
    features = np.array([[0.0], [1.0], [2.0], [10.0]])
    feature_index = np.array([3, 1, 2, 0])

    selected = manifests.select_narrow(features, feature_index, 2)

    # mean = 3.25; distances 3.25, 2.25, 1.25, 6.75 -> nearest two are rows 2 and 1.
    assert set(selected.tolist()) == {2, 1}


def test_select_wide_seeds_at_mean_and_grows_farthest_first() -> None:
    features = np.array([[0.0], [1.0], [2.0], [10.0]])
    feature_index = np.array([0, 1, 2, 3])

    selected = manifests.select_wide(features, feature_index, 2)

    # mean = 3.25 -> nearest is row 2 (dist 1.25), seeded first; farthest
    # remaining from row 2 is row 3 (dist 8.0).
    assert selected.tolist() == [2, 3]


def test_select_wide_returns_everything_when_headroom_is_exactly_one() -> None:
    features = np.array([[0.0], [1.0], [2.0]])
    feature_index = np.array([0, 1, 2])

    selected = manifests.select_wide(features, feature_index, 3)

    assert sorted(selected.tolist()) == [0, 1, 2]


def test_build_allocation_levels_pins_counts_case_and_slide_sets(tmp_path: Path) -> None:
    _, source_df = _synthetic_manifest(tmp_path)
    freeze = _exp2_freeze(source_df)
    out_dir = tmp_path / "exp3"
    out_dir.mkdir()

    result = manifests.build_allocation_levels(
        "balanced", tmp_path, out_dir, freeze, CLASS_NAMES
    )

    dataframes = result["dataframes"]
    reference_counts = (
        dataframes["random"].groupby(["cancer_type", "case_id", "slide_id"]).size()
    )
    for level in manifests.LEVELS:
        counts = dataframes[level].groupby(["cancer_type", "case_id", "slide_id"]).size()
        pd.testing.assert_series_equal(
            counts.sort_index(), reference_counts.sort_index(), check_names=False
        )


def test_build_allocation_levels_identical_case_and_slide_id_sets(tmp_path: Path) -> None:
    _, source_df = _synthetic_manifest(tmp_path)
    freeze = _exp2_freeze(source_df)
    out_dir = tmp_path / "exp3"
    out_dir.mkdir()

    result = manifests.build_allocation_levels(
        "balanced", tmp_path, out_dir, freeze, CLASS_NAMES
    )

    dataframes = result["dataframes"]
    for key in ("case_id", "slide_id"):
        reference = set(dataframes["random"][key])
        for level in manifests.LEVELS:
            assert set(dataframes[level][key]) == reference


def test_build_allocation_levels_identical_contribution_stats(tmp_path: Path) -> None:
    _, source_df = _synthetic_manifest(tmp_path)
    freeze = _exp2_freeze(source_df)
    out_dir = tmp_path / "exp3"
    out_dir.mkdir()

    result = manifests.build_allocation_levels(
        "balanced", tmp_path, out_dir, freeze, CLASS_NAMES
    )

    conditions = result["conditions"]
    reference = conditions["random"]["contribution_stats"]
    for level in manifests.LEVELS:
        assert conditions[level]["contribution_stats"] == reference
        assert conditions[level]["allocated_counts"] == conditions["random"]["allocated_counts"]


def test_build_allocation_levels_narrow_differs_from_wide_when_headroom_exceeds_one(
    tmp_path: Path,
) -> None:
    _, source_df = _synthetic_manifest(tmp_path)
    freeze = _exp2_freeze(source_df)
    out_dir = tmp_path / "exp3"
    out_dir.mkdir()

    result = manifests.build_allocation_levels(
        "balanced", tmp_path, out_dir, freeze, CLASS_NAMES
    )

    # Headroom here is POOL_SIZE / PINNED_N = 3 > 1 for every slot.
    assert (result["headroom"]["h"] > 1.0).all()
    narrow_ids = set(result["dataframes"]["narrow"]["patch_id"])
    wide_ids = set(result["dataframes"]["wide"]["patch_id"])
    assert narrow_ids != wide_ids


def test_build_allocation_levels_is_deterministic_across_two_invocations(
    tmp_path: Path,
) -> None:
    _, source_df = _synthetic_manifest(tmp_path)
    freeze = _exp2_freeze(source_df)

    out_dir_1 = tmp_path / "exp3_run1"
    out_dir_1.mkdir()
    result_1 = manifests.build_allocation_levels(
        "balanced", tmp_path, out_dir_1, freeze, CLASS_NAMES
    )

    out_dir_2 = tmp_path / "exp3_run2"
    out_dir_2.mkdir()
    result_2 = manifests.build_allocation_levels(
        "balanced", tmp_path, out_dir_2, freeze, CLASS_NAMES
    )

    for level in manifests.LEVELS:
        pd.testing.assert_frame_equal(
            result_1["dataframes"][level].reset_index(drop=True),
            result_2["dataframes"][level].reset_index(drop=True),
        )
        assert (
            result_1["conditions"][level]["contribution_stats"]
            == result_2["conditions"][level]["contribution_stats"]
        )


def test_build_allocation_levels_raises_on_slot_count_violation(
    tmp_path: Path, monkeypatch
) -> None:
    """A construction bug that breaks pinning must raise, never warn."""
    _, source_df = _synthetic_manifest(tmp_path)
    freeze = _exp2_freeze(source_df)
    out_dir = tmp_path / "exp3"
    out_dir.mkdir()

    original_select_narrow = manifests_allocation._select_narrow

    def _broken_select_narrow(features, feature_index, n):
        return original_select_narrow(features, feature_index, max(0, n - 1))

    monkeypatch.setattr(manifests_allocation, "_select_narrow", _broken_select_narrow)

    try:
        manifests.build_allocation_levels("balanced", tmp_path, out_dir, freeze, CLASS_NAMES)
    except RuntimeError as error:
        assert "counts differ" in str(error)
    else:
        raise AssertionError("Expected a RuntimeError for a broken pinned count")


def test_build_derived_freeze_replaces_only_assignment_axes() -> None:
    exp2_freeze = {
        "class_names": CLASS_NAMES,
        "min_support": PINNED_N,
        "seed_roles": {"confirmation_initialization_0": 7},
        "tail_assignments": {"native": ["A", "B"], "difficulty_aligned": ["B", "A"]},
        "assignment_conditions": {"native": {"severe": {"allocated_counts": {"A": 1}}}},
        "conditions": {"balanced": {"allocated_counts": {"A": 1, "B": 1}}},
        "runtime_config": {"dataset": {"name": "synthetic"}},
    }
    level_conditions = {
        "narrow": {"balanced": {"allocated_counts": {"A": 1, "B": 1}}, "severe": {}},
        "random": {"balanced": {"allocated_counts": {"A": 1, "B": 1}}, "severe": {}},
        "wide": {"balanced": {"allocated_counts": {"A": 1, "B": 1}}, "severe": {}},
    }

    derived = manifests.build_derived_freeze(exp2_freeze, level_conditions)

    assert derived["assignment_conditions"] == level_conditions
    assert derived["tail_assignments"] == {
        "narrow": ["A", "B"],
        "random": ["A", "B"],
        "wide": ["A", "B"],
    }
    # Untouched keys survive the deep copy unchanged.
    assert derived["runtime_config"] == exp2_freeze["runtime_config"]
    assert derived["conditions"] == exp2_freeze["conditions"]
    assert derived["seed_roles"] == exp2_freeze["seed_roles"]
    assert "content_sha256" in derived
