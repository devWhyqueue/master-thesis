from __future__ import annotations

import pandas as pd
import pytest

from imbalance_benchmark.construction import (
    allocate_counts,
    patient_equals_slide,
    select_patches_round_robin,
    select_slides_round_robin,
    validate_split_leakage,
)
from imbalance_benchmark.manifest.freeze import (
    achieved_rho,
    build_tail_assignments,
    contribution_stats,
    normalized_entropy,
    verify_manifest_freeze,
)
from imbalance_benchmark.manifest.pilot import (
    build_patch_pilot_manifest,
    compute_pilot_quota,
    method_floor,
    pilot_levels_for,
    stability_floor_from_curve,
)
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.commands.freeze import _build_conditions


def _patch_frame(n_patients: int, slides_per_patient: int, patches_per_slide: int) -> pd.DataFrame:
    rows = []
    for p in range(n_patients):
        for s in range(slides_per_patient):
            for patch in range(patches_per_slide):
                rows.append(
                    {
                        "case_id": f"PAT_{p}",
                        "slide_id": f"PAT_{p}_SLIDE_{s}",
                        "patch_id": f"PAT_{p}_SLIDE_{s}_PATCH_{patch}",
                        "cancer_type": "class_A",
                    }
                )
    return pd.DataFrame(rows)


def test_validate_split_leakage_raises_on_case_in_two_splits():
    df = pd.DataFrame(
        {
            "case_id": ["PAT_0", "PAT_0", "PAT_1"],
            "split": ["train", "test", "train"],
        }
    )
    with pytest.raises(RuntimeError, match="PAT_0"):
        validate_split_leakage(df)


def test_validate_split_leakage_passes_when_disjoint():
    df = pd.DataFrame(
        {
            "case_id": ["PAT_0", "PAT_1", "PAT_2"],
            "split": ["train", "test", "validation"],
        }
    )
    validate_split_leakage(df)  # must not raise


def test_patient_equals_slide_detection():
    one_to_one = pd.DataFrame({"case_id": ["A", "B"], "slide_id": ["A", "B"]})
    one_to_many = pd.DataFrame({"case_id": ["A", "A", "B"], "slide_id": ["A1", "A2", "B1"]})
    assert patient_equals_slide(one_to_one)
    assert not patient_equals_slide(one_to_many)


def test_deterministic_patch_selection_same_seed():
    df = _patch_frame(n_patients=20, slides_per_patient=3, patches_per_slide=5)
    first = select_patches_round_robin(df, 30, seed=7)
    second = select_patches_round_robin(df, 30, seed=7)
    assert list(first["patch_id"]) == list(second["patch_id"])


def test_patch_selection_respects_patient_and_slide_caps():
    df = _patch_frame(n_patients=20, slides_per_patient=4, patches_per_slide=5)
    n_patches = 40
    selected = select_patches_round_robin(df, n_patches, seed=3)
    patient_cap = max(1, round(n_patches * 0.10))
    slide_cap = max(1, round(n_patches * 0.05))
    assert (selected["case_id"].value_counts() <= patient_cap).all()
    assert (selected["slide_id"].value_counts() <= slide_cap).all()


def test_slide_selection_respects_patient_cap():
    rows = []
    for p in range(20):
        for s in range(3):
            rows.append(
                {
                    "case_id": f"PAT_{p}",
                    "slide_id": f"PAT_{p}_SLIDE_{s}",
                    "cancer_type": "class_A",
                }
            )
    df = pd.DataFrame(rows)
    n_slides = 30
    selected = select_slides_round_robin(df, n_slides, seed=5)
    patient_cap = max(1, round(n_slides * 0.10))
    unique_slides = selected.drop_duplicates("slide_id")
    assert (unique_slides["case_id"].value_counts() <= patient_cap).all()


def test_allocate_counts_feasibility_fallback_when_available_is_tight():
    available = [12, 12, 12, 12]
    counts = allocate_counts(available, total_t=48, rho=100.0, min_support=10)
    assert sum(counts) == 48
    assert all(c >= 10 for c in counts)
    assert all(c <= a for c, a in zip(counts, available))
    # Requested rho=100 is infeasible under a floor of 10 on a pool of 12;
    # the achieved ratio must fall well short of the request.
    assert achieved_rho(dict(enumerate(counts))) < 100.0


def test_normalized_entropy_zero_when_balanced():
    assert normalized_entropy([10, 10, 10, 10]) == pytest.approx(0.0, abs=1e-9)


def test_normalized_entropy_increases_with_skew():
    balanced = normalized_entropy([25, 25, 25, 25])
    moderate = normalized_entropy([70, 10, 10, 10])
    severe = normalized_entropy([97, 1, 1, 1])
    assert balanced < moderate < severe


def test_achieved_rho_matches_max_over_min():
    assert achieved_rho({"a": 100, "b": 10, "c": 50}) == pytest.approx(10.0)
    assert achieved_rho({"a": 0, "b": 0}) == 1.0


def test_build_tail_assignments_variants_and_reproducibility():
    classes = ["class_A", "class_B", "class_C", "class_D"]
    assignments = build_tail_assignments(classes, seed=11, ordinal=False)
    assert assignments["native"] == classes
    assert assignments["reversed_or_rotated"] == classes[1:] + classes[:1]
    assert sorted(assignments["random"]) == sorted(classes)
    again = build_tail_assignments(classes, seed=11, ordinal=False)
    assert assignments["random"] == again["random"]
    different_seed = build_tail_assignments(classes, seed=12, ordinal=False)
    assert different_seed["random"] != assignments["random"]


def test_build_tail_assignments_reversed_for_ordinal():
    classes = ["g0", "g1", "g2", "g3"]
    assignments = build_tail_assignments(classes, seed=1, ordinal=True)
    assert assignments["reversed_or_rotated"] == list(reversed(classes))


def test_contribution_stats_reports_pool_fraction():
    pool = _patch_frame(n_patients=10, slides_per_patient=2, patches_per_slide=5)
    condition = select_patches_round_robin(pool, 20, seed=1)
    stats = contribution_stats(condition, pool, is_mil=False)
    assert stats["class_A"]["pool_fraction_retained"] <= 1.0
    assert stats["class_A"]["n_patches"] == 20


def test_patch_conditions_are_nested_prefixes_of_one_frozen_master_pool(tmp_path):
    pool = pd.concat(
        [
            _patch_frame(20, 3, 10).assign(cancer_type="class_A"),
            _patch_frame(20, 3, 10).assign(cancer_type="class_B"),
        ],
        ignore_index=True,
    )
    pool["case_id"] = pool["cancer_type"] + "_" + pool["case_id"]
    pool["slide_id"] = pool["cancer_type"] + "_" + pool["slide_id"]
    pool["split"] = "train"
    conditions = _build_conditions(
        pool, ["class_A", "class_B"], 200, 10, False, 3, tmp_path
    )
    balanced = pd.read_csv(conditions["balanced"]["path"])
    severe = pd.read_csv(conditions["severe"]["path"])
    for cls in ["class_A", "class_B"]:
        balanced_ids = balanced[balanced["cancer_type"] == cls]["patch_id"].tolist()
        severe_ids = severe[severe["cancer_type"] == cls]["patch_id"].tolist()
        assert set(severe_ids) <= set(balanced_ids) or set(balanced_ids) <= set(severe_ids)


def test_verify_manifest_freeze_detects_tampering(tmp_path):
    path = tmp_path / "manifest_balanced.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    from imbalance_benchmark.common import compute_sha256

    meta = {"conditions": {"balanced": {"path": str(path), "sha256": compute_sha256(path)}}}
    verify_manifest_freeze(meta)  # must not raise
    pd.DataFrame({"a": [1, 2, 999]}).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="balanced"):
        verify_manifest_freeze(meta)


def test_derive_seed_keeps_families_disjoint_and_deterministic():
    base = 42
    pilot_seeds = [derive_seed(base, f"pilot_construction_{i}") for i in range(3)]
    definitive_seed = derive_seed(base, "definitive_construction")
    assert len(set(pilot_seeds)) == 3
    assert definitive_seed not in pilot_seeds
    assert derive_seed(base, "patient_split") != derive_seed(base, "assignment")
    assert derive_seed(base, "pilot_construction_0") == derive_seed(base, "pilot_construction_0")
    with pytest.raises(ValueError):
        derive_seed(base, "not_a_real_role")


def test_pilot_levels_for_caps_at_scarcest_class():
    levels = pilot_levels_for({"class_A": 40, "class_B": 8})
    assert levels == [5, 8]


def test_pilot_levels_for_uses_standard_candidates_when_plentiful():
    levels = pilot_levels_for({"class_A": 100, "class_B": 100})
    assert levels == [5, 10, 15, 20, 30]


def test_compute_pilot_quota_is_feasible_for_every_class():
    df = pd.concat(
        [
            _patch_frame(15, 2, 3).assign(cancer_type="class_A"),
            _patch_frame(15, 2, 6).assign(cancer_type="class_B"),
        ],
        ignore_index=True,
    )
    # Disambiguate identical case_id/slide_id across the two concatenated classes.
    df["case_id"] = df["cancer_type"] + "_" + df["case_id"]
    df["slide_id"] = df["cancer_type"] + "_" + df["slide_id"]
    quota = compute_pilot_quota(df, ["class_A", "class_B"], level=10, seed=1)
    assert quota >= 1
    # class_A has only 3 patches/slide x 2 slides = 6 patches per patient.
    assert quota <= 6


def test_build_patch_pilot_manifest_respects_quota_per_patient():
    df = _patch_frame(n_patients=10, slides_per_patient=2, patches_per_slide=10)
    manifest = build_patch_pilot_manifest(df, ["class_A"], level=5, quota=4, seed=2)
    assert manifest["case_id"].nunique() == 5
    assert (manifest["case_id"].value_counts() == 4).all()


def test_method_floor_collapses_when_patient_equals_slide():
    assert method_floor(patient_equals_slide=True) == {"slides": 20}
    assert method_floor(patient_equals_slide=False) == {"patients": 10, "slides": 20}


def test_stability_floor_falls_back_to_largest_when_never_stable():
    levels = [5, 10, 15]
    ba = {0: [0.5, 0.7, 0.95]}
    recalls = {0: [[0.5, 0.5], [0.7, 0.7], [0.95, 0.95]]}
    assert stability_floor_from_curve(levels, ba, recalls) == 15


def test_stability_floor_picks_first_stable_level():
    levels = [5, 10, 15]
    ba = {0: [0.5, 0.55, 0.551]}
    recalls = {0: [[0.5, 0.5], [0.55, 0.55], [0.551, 0.552]]}
    assert stability_floor_from_curve(levels, ba, recalls) == 10
