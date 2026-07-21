from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from imbalance_benchmark.analysis.inference.context import _tail_classes
from imbalance_benchmark.analysis.reporting.plots import (
    allocated_training_support,
    plot_tail_vs_support,
)
from imbalance_benchmark.commands.freeze import _build_conditions
from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.construction import (
    allocate_counts,
    effective_rho,
    max_shared_total,
    select_patches_round_robin,
)
from imbalance_benchmark.construction import (
    patient_equals_slide,
    select_slides_round_robin,
    validate_split_leakage,
)
from imbalance_benchmark.construction import _allocation_is_feasible
from imbalance_benchmark.manifest.construction_helpers import write_natural_condition
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
    mil_pilot_manifest,
    pilot_levels_for,
    stability_floor_from_curve,
)
from imbalance_benchmark.manifest.seeds import derive_seed

def _patch_frame(
    n_patients: int, slides_per_patient: int, patches_per_slide: int
) -> pd.DataFrame:
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

def _patches(class_name: str, n_patients: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                f"{class_name}_patient_{patient}"
                for patient in range(n_patients)
                for _ in range(10)
            ],
            "slide_id": [
                f"{class_name}_patient_{patient}_slide_{patch % 2}"
                for patient in range(n_patients)
                for patch in range(10)
            ],
            "patch_id": [
                f"{class_name}_{patient}_{patch}"
                for patient in range(n_patients)
                for patch in range(10)
            ],
            "cancer_type": class_name,
            "split": "train",
        }
    )

def test_natural_anchor_records_its_support_and_contribution_statistics(
    tmp_path: Path,
) -> None:
    """The descriptive anchor must remain fully auditable alongside controls."""
    rows = pd.DataFrame(
        [
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p2", "slide_id": "s2", "cancer_type": "B"},
        ]
    )

    natural = write_natural_condition(rows, tmp_path, is_mil=False)

    assert natural["allocated_counts"] == {"A": 2, "B": 1}
    assert natural["achieved_rho"] == 2.0
    assert natural["contribution_stats"]["A"]["pool_fraction_retained"] == 1.0

def test_mil_slide_contribution_uses_one_slide_one_example() -> None:
    """Patch-row multiplicity must not inflate a WSI slide's contribution."""
    rows = pd.DataFrame(
        [
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p2", "slide_id": "s2", "cancer_type": "A"},
        ]
    )

    stats = contribution_stats(rows, rows, is_mil=True)

    assert stats["A"]["max_slide_contribution"] == 0.5

def test_effective_rho_finds_narrow_highest_feasible_interval() -> None:
    available = [75928, 168239, 36174, 130815, 104503, 32398, 127285]

    result = effective_rho(available, 100.0, 30, 226792)

    assert result == pytest.approx(9.251803719617433, rel=1e-10)

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
    one_to_many = pd.DataFrame(
        {"case_id": ["A", "A", "B"], "slide_id": ["A1", "A2", "B1"]}
    )
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

def test_natural_mil_statistics_are_reported_at_slide_and_patch_levels(tmp_path):
    """WSI anchors count allocated support by slides while retaining patch statistics."""
    rows = pd.DataFrame(
        [
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p2", "slide_id": "s2", "cancer_type": "A"},
            {"case_id": "p3", "slide_id": "s3", "cancer_type": "B"},
        ]
    )

    natural = write_natural_condition(rows, tmp_path, is_mil=True)

    assert natural["allocated_counts"] == {"A": 2, "B": 1}
    assert natural["support_statistics"]["patch"]["counts"] == {"A": 3, "B": 1}
    assert natural["support_statistics"]["slide"]["counts"] == {"A": 2, "B": 1}

def test_patch_conditions_respect_caps_at_each_condition_size(tmp_path):
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
        pool, ["class_A", "class_B"], 440, 20, False, 3, tmp_path
    )
    for condition in conditions.values():
        frame = pd.read_csv(condition["path"])
        for cls, n_patches in condition["allocated_counts"].items():
            rows = frame[frame["cancer_type"] == cls]
            assert pd.Series(rows["case_id"]).value_counts().max() <= int(
                n_patches * 0.10
            )
            assert pd.Series(rows["slide_id"]).value_counts().max() <= int(
                n_patches * 0.05
            )

def test_verify_manifest_freeze_detects_tampering(tmp_path):
    path = tmp_path / "manifest_balanced.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)

    meta = {
        "conditions": {"balanced": {"path": str(path), "sha256": compute_sha256(path)}}
    }
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
    assert derive_seed(base, "pilot_construction_0") == derive_seed(
        base, "pilot_construction_0"
    )
    with pytest.raises(ValueError):
        derive_seed(base, "not_a_real_role")

def test_pilot_levels_for_caps_at_scarcest_class():
    levels = pilot_levels_for({"class_A": 40, "class_B": 8})
    assert levels == [8]

def test_pilot_levels_for_uses_standard_candidates_when_plentiful():
    levels = pilot_levels_for({"class_A": 100, "class_B": 100})
    assert levels == [10, 15, 20, 30, 50]

def test_mil_pilot_smallest_level_satisfies_the_patient_cap_without_exception():
    """No hardcoded small-count exception: level 10 already satisfies the 10% cap."""
    rows = [
        {"case_id": f"PAT_{index}", "slide_id": f"SLIDE_{index}", "cancer_type": "A"}
        for index in range(10)
    ]

    manifest = mil_pilot_manifest(pd.DataFrame(rows), ["A"], level=10, seed=1)

    assert len(manifest) == 10
    assert manifest["case_id"].nunique() == 10

def test_mil_pilot_level_five_raises_without_the_removed_exception():
    """Five slides from five distinct patients unavoidably breach the 10% cap."""
    rows = [
        {"case_id": f"PAT_{index}", "slide_id": f"SLIDE_{index}", "cancer_type": "A"}
        for index in range(10)
    ]

    with pytest.raises(ValueError, match="patient cap"):
        mil_pilot_manifest(pd.DataFrame(rows), ["A"], level=5, seed=1)

def test_patch_pilot_level_five_raises_without_the_removed_exception():
    """Five equal patient quotas unavoidably imply 20% patient support; with the
    small-count exception removed, this now raises instead of being carved out."""
    rows = [
        {
            "case_id": f"PAT_{patient}",
            "slide_id": f"PAT_{patient}_SLIDE_{slide}",
            "patch_id": f"PAT_{patient}_SLIDE_{slide}_PATCH_{patch}",
            "cancer_type": "class_A",
        }
        for patient in range(10)
        for slide in range(4)
        for patch in range(2)
    ]
    frame = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="cannot satisfy"):
        compute_pilot_quota(frame, ["class_A"], level=5, seed=0)

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
    manifest = build_patch_pilot_manifest(df, ["class_A"], level=10, quota=4, seed=2)
    assert manifest["case_id"].nunique() == 10
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

def test_tail_support_plot_uses_frozen_training_allocation() -> None:
    classwise = pd.DataFrame(
        [
            {
                "assignment": "native",
                "condition": "severe",
                "class_name": "A",
                "support": 99,
            }
        ]
    )
    freeze = {
        "assignment_conditions": {"native": {"severe": {"allocated_counts": {"A": 7}}}}
    }

    assert allocated_training_support(classwise, freeze).tolist() == [7]

def test_tail_support_plot_excludes_conditions_without_a_frozen_tier(
    tmp_path: Path,
) -> None:
    classwise = pd.DataFrame(
        [
            {
                "assignment": "native",
                "condition": "natural",
                "class_name": "A",
                "tier": None,
                "support": 99,
                "recall": 0.5,
            },
            {
                "assignment": "native",
                "condition": "severe",
                "class_name": "A",
                "tier": "tail",
                "support": 99,
                "recall": 0.5,
            },
        ]
    )
    freeze = {
        "assignment_conditions": {"native": {"severe": {"allocated_counts": {"A": 7}}}}
    }

    plot_tail_vs_support(classwise, freeze, tmp_path / "tail.png")

    assert (tmp_path / "tail.png").exists()

def test_balanced_reference_uses_the_largest_approximately_equal_total() -> None:
    assert max_shared_total([10, 10, 11], min_support=5) == 31

def test_asymmetric_availability_keeps_the_largest_approximately_balanced_total() -> (
    None
):
    available = [1000, 500, 200]

    total = max_shared_total(available, min_support=20)
    rho = effective_rho(available, rho=100.0, min_support=20, total_t=total)
    allocation = allocate_counts(available, total, rho, min_support=20)

    assert total == 602
    assert 1.0 < rho < 100.0
    assert sum(allocation) == total
    assert min(allocation) >= 20
    assert all(
        count <= capacity for count, capacity in zip(allocation, available, strict=True)
    )

def test_evidence_seed_is_stable_when_a_semantic_class_changes_tail_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = pd.concat([_patches("A"), _patches("B")], ignore_index=True)
    observed: list[tuple[str, int]] = []

    def selector(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        observed.append((str(df["cancer_type"].iloc[0]), seed))
        return df.iloc[:n]

    monkeypatch.setattr(
        "imbalance_benchmark.manifest.freezing.select_patches_round_robin", selector
    )
    _build_conditions(
        frame, ["A", "B"], 100, 20, False, 17, tmp_path, condition_names=("moderate",)
    )
    first = dict(observed)
    observed.clear()
    _build_conditions(
        frame, ["B", "A"], 100, 20, False, 17, tmp_path, condition_names=("moderate",)
    )

    assert dict(observed) == first

def test_patch_conditions_record_one_fixed_patient_slide_pool(tmp_path: Path) -> None:
    frame = pd.concat([_patches("A"), _patches("B")], ignore_index=True)

    conditions = _build_conditions(frame, ["A", "B"], 80, 20, False, 8, tmp_path)

    assert (
        conditions["balanced"]["evidence_pool_hash"]
        == conditions["moderate"]["evidence_pool_hash"]
    )
    assert (
        conditions["moderate"]["evidence_pool_hash"]
        == conditions["severe"]["evidence_pool_hash"]
    )

def test_smaller_patch_allocation_is_a_nested_subset_of_the_larger_pool() -> None:
    """A smaller allocation must reuse the larger one's fixed patient/slide pool.

    Breadth-first round-robin makes every smaller patch allocation a nested
    prefix of the larger one: its patients and slides are subsets and patient
    diversity is preserved, rather than concentrating into a few units.
    """
    df_class = _patches(
        "A", n_patients=20
    )  # 20 patients, 2 slides each, 10 patches each

    small = select_patches_round_robin(df_class, 20, seed=5)
    large = select_patches_round_robin(df_class, 40, seed=5)

    # Every pool patient contributes before any patch is doubled up: 20 patches
    # spread one-per-patient over all 20 patients rather than 10 patients x 2.
    assert small["case_id"].nunique() == 20
    assert set(small.index) <= set(large.index)
    assert set(small["case_id"]) <= set(large["case_id"])
    assert set(small["slide_id"]) <= set(large["slide_id"])
    assert small["case_id"].nunique() == large["case_id"].nunique()

def test_patch_conditions_use_the_same_designated_patient_and_slide_pools(
    tmp_path: Path,
) -> None:
    """Controlled patch conditions must retain one explicit, identical evidence pool."""
    frame = pd.concat([_patches("A", 30), _patches("B", 30)], ignore_index=True)

    conditions = _build_conditions(frame, ["A", "B"], 80, 20, False, 4, tmp_path)
    pools = {
        name: pd.read_csv(info["path"])
        .groupby("cancer_type")[["case_id", "slide_id"]]
        .agg(lambda values: frozenset(values))
        for name, info in conditions.items()
    }

    # The fixed design pool is retained in every condition, rather than merely
    # making smaller selections nested subsets of the larger one.
    assert pools["balanced"].equals(pools["moderate"])
    assert pools["balanced"].equals(pools["severe"])

def test_patch_conditions_retain_every_independent_unit_in_the_fixed_pool(
    tmp_path: Path,
) -> None:
    """Every condition keeps the pilot-required patients and slides, not just its hash."""
    frame = pd.concat([_patches("A", 30), _patches("B", 30)], ignore_index=True)

    conditions = _build_conditions(
        frame,
        ["A", "B"],
        shared_t=200,
        min_support=60,
        is_mil=False,
        seed=4,
        data_dir=tmp_path,
        independent_floor=30,
    )

    for info in conditions.values():
        stats = info["contribution_stats"]
        assert all(entry["n_patients"] >= 30 for entry in stats.values())
        assert all(entry["n_slides"] >= 30 for entry in stats.values())
        assert all(entry["pool_fraction_retained"] == 1.0 for entry in stats.values())

def test_fixed_pool_expansion_adds_one_slide_per_patient_per_round() -> None:
    """A fixed evidence pool expands breadth-first over its selected patients."""
    from imbalance_benchmark.manifest.construction_sampling import designate_patch_pool

    rows = []
    for patient in range(10):
        n_slides = 3 if patient == 0 else 2
        for slide in range(n_slides):
            for patch in range(10):
                rows.append(
                    {
                        "case_id": f"patient_{patient}",
                        "slide_id": f"patient_{patient}_slide_{slide}",
                        "patch_id": f"patient_{patient}_{slide}_{patch}",
                        "cancer_type": "A",
                        "split": "train",
                    }
                )
    pool = designate_patch_pool(
        pd.DataFrame(rows), 10, seed=4, max_p=180, max_pool_units=20
    )

    assert pool.groupby("case_id")["slide_id"].nunique().eq(2).all()

@pytest.mark.parametrize("seed", range(60))
def test_random_tail_assignment_is_distinct_from_native_and_rotated(seed: int) -> None:
    """The random permutation must not duplicate the native or rotated assignment."""
    assignments = build_tail_assignments(["A", "B", "C"], seed=seed, ordinal=False)

    orders = [tuple(order) for order in assignments.values()]
    assert len(set(orders)) == 3

def test_effective_rho_returns_largest_feasible_when_feasibility_is_disconnected():
    inventory = [372, 231, 107, 463, 114, 364, 96]
    floor, total = 20, 678
    # rho=10 is demonstrably feasible, so a severe (rho=100) request must not
    # collapse below the moderate rho=10 it already attains.
    assert _allocation_is_feasible(inventory, total, 10.0, floor)
    severe = effective_rho(inventory, 100.0, floor, total)
    moderate = effective_rho(inventory, 10.0, floor, total)
    assert severe >= 10.0
    assert severe >= moderate

def test_stability_floor_requires_ba_increment_below_threshold_in_every_ordering():
    levels = [5, 10]
    # Mean BA increment is zero (+0.015, -0.015, 0), but two orderings each
    # exceed the 0.01 criterion, so level 5 is not stable.
    ba = {0: [0.5, 0.515], 1: [0.5, 0.485], 2: [0.5, 0.5]}
    recalls = {
        0: [[0.5, 0.5], [0.505, 0.505]],
        1: [[0.5, 0.5], [0.5, 0.5]],
        2: [[0.5, 0.5], [0.5, 0.5]],
    }
    assert stability_floor_from_curve(levels, ba, recalls) == 10

def test_tail_classes_follow_the_analysed_severity_allocation():
    names = ["A", "B", "C", "D"]
    freeze = {
        "assignment_conditions": {
            "native": {
                "moderate": {"allocated_counts": {"A": 100, "B": 80, "C": 20, "D": 10}},
                "severe": {"allocated_counts": {"A": 100, "B": 10, "C": 80, "D": 90}},
            }
        },
        "tail_assignments": {"native": names},
    }
    moderate = _tail_classes(freeze, names, "native", "moderate")
    severe = _tail_classes(freeze, names, "native", "severe")
    assert moderate == [2, 3]  # C, D are the scarcest under moderate
    assert severe == [1, 2]  # B, C are the scarcest under severe
    assert moderate != severe
