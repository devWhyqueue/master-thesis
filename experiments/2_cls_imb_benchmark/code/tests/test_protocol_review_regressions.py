from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imbalance_benchmark.analysis.inference.crossed_permutation import (
    crossed_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.inference.preflight import run_preflight
from imbalance_benchmark.analysis.query import _confirmation_dir, load_seed_predictions
from imbalance_benchmark.commands.pilot import _pilot_report_payload
from imbalance_benchmark.common import dataset_provenance, write_run_record
from imbalance_benchmark.construction import (
    allocate_counts,
    effective_rho,
    max_shared_total,
    select_patches_round_robin,
)
from imbalance_benchmark.manifest.freeze import build_tail_assignments
from imbalance_benchmark.manifest.freezing import _build_conditions
from imbalance_benchmark.manifest.pilot import (
    build_patch_pilot_manifest,
    compute_pilot_quota,
    frozen_pilot_quota,
    meets_method_floor,
)


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

    conditions = _build_conditions(frame, ["A", "B"], 80, 20, False, 8, tmp_path)

    assert conditions["balanced"]["evidence_pool_hash"] == conditions["moderate"]["evidence_pool_hash"]
    assert conditions["moderate"]["evidence_pool_hash"] == conditions["severe"]["evidence_pool_hash"]


def test_smaller_patch_allocation_is_a_nested_subset_of_the_larger_pool() -> None:
    """A smaller allocation must reuse the larger one's fixed patient/slide pool.

    Breadth-first round-robin makes every smaller patch allocation a nested
    prefix of the larger one: its patients and slides are subsets and patient
    diversity is preserved, rather than concentrating into a few units.
    """
    df_class = _patches("A", n_patients=20)  # 20 patients, 2 slides each, 10 patches each

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

    conditions = _build_conditions(
        frame, ["A", "B"], 80, 20, False, 4, tmp_path
    )
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


def test_freeze_uses_one_patch_pool_for_balanced_and_every_assignment(
    tmp_path: Path,
) -> None:
    """A valid reversed assignment must reuse the balanced class pools."""
    from argparse import Namespace

    from imbalance_benchmark.manifest.freezing import _freeze_meta

    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset:\n  name: synthetic\n  regime: patch\n  target: diagnosis\n",
        encoding="utf-8",
    )
    rows = []
    for class_name in ("A", "B"):
        for patient in range(10):
            n_slides = 3 if patient == 0 else 2
            for slide in range(n_slides):
                for patch in range(10):
                    rows.append(
                        {
                            "case_id": f"{class_name}_{patient}",
                            "slide_id": f"{class_name}_{patient}_{slide}",
                            "patch_id": f"{class_name}_{patient}_{slide}_{patch}",
                            "cancer_type": class_name,
                            "split": "train",
                        }
                    )
    meta = _freeze_meta(
        Namespace(seed=4, config=config),
        {"data": tmp_path},
        pd.DataFrame(rows),
        False,
        ["A", "B"],
        200,
        20,
        20,
        False,
        10,
    )

    pool_hashes = {
        info["evidence_pool_hash"]
        for conditions in [meta["conditions"], *meta["assignment_conditions"].values()]
        for info in conditions.values()
    }
    assert len(pool_hashes) == 1


def test_freeze_rejects_missing_dataset_provenance(tmp_path: Path) -> None:
    """Definitive freezes cannot replace required provenance with placeholders."""
    from imbalance_benchmark.commands.freeze_execution import _attach_provenance

    pilot = tmp_path / "pilot_report.json"
    manifest = tmp_path / "manifest.csv"
    pilot.write_text("{}", encoding="utf-8")
    manifest.write_text("case_id,split\nA,train\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset.version"):
        _attach_provenance(
            {},
            {"data": tmp_path},
            {"dataset": {"name": "synthetic", "regime": "patch"}},
        )


def test_dataset_provenance_requires_a_frozen_target() -> None:
    dataset = {
        "name": "panda",
        "regime": "wsi",
        "version": "v1",
        "eligibility_rules": {"slide_qc": "pass"},
    }

    with pytest.raises(ValueError, match="dataset.target"):
        dataset_provenance(dataset)

    provenance = dataset_provenance({**dataset, "target": "isup_grade"})

    assert provenance["target"] == "isup_grade"


def test_pilot_definitive_floor_does_not_collapse_patient_and_slide_floors() -> None:
    """Patch pilot levels count patients; the slide floor must not become a 20-patient floor."""
    levels = [5, 10, 15, 20, 30]
    flat_ba = {seed: [0.5] * len(levels) for seed in (0, 1, 2)}
    flat_recall = {seed: [[0.5, 0.5]] * len(levels) for seed in (0, 1, 2)}
    support = {"A": {"patients": 12, "slides": 25}, "B": {"patients": 12, "slides": 25}}

    patch = _pilot_report_payload(
        levels, False, False, [0, 1, 2], {}, flat_ba, flat_recall, support
    )
    mil = _pilot_report_payload(
        levels, True, False, [0, 1, 2], {}, flat_ba, flat_recall, support
    )

    # Patch pilot counts patients -> patient floor 10, not the 20-slide floor.
    assert patch["stability_floor"] == 5
    assert patch["definitive_floor"] == 10
    assert patch["excluded"] is False
    # MIL pilot counts slides -> slide floor 20 applies to the level dimension.
    assert mil["definitive_floor"] == 20
    assert mil["pilot_exceptions"] == [
        "five-slide MIL pilot uses one slide from each of five distinct patients"
    ]


def test_patch_pilot_patient_floor_is_preserved_as_an_independent_constraint(
    tmp_path: Path,
) -> None:
    """A patient floor cannot be weakened into an equivalent patch count."""
    from imbalance_benchmark.manifest.freezing import _pilot_constraints

    report = tmp_path / "pilot_report.json"
    report.write_text(
        json.dumps({"definitive_floor": 30, "quotas": {"1": 2}, "excluded": False}),
        encoding="utf-8",
    )

    constraints = _pilot_constraints(report, is_mil=False)

    assert constraints.patch_floor == 60
    assert constraints.independent_floor == 30


def test_pilot_quota_is_frozen_across_construction_orderings() -> None:
    """One quota is shared by every pilot ordering, not recomputed per seed."""
    patients = []
    for patient in range(12):
        n_patches = 4 + patient  # varying inventory so per-seed selection matters
        patients.append(
            pd.DataFrame(
                {
                    "case_id": [f"A_{patient}"] * n_patches,
                    "slide_id": [f"A_{patient}_s{p % 2}" for p in range(n_patches)],
                    "cancer_type": "A",
                    "split": "train",
                }
            )
        )
    df = pd.concat(patients, ignore_index=True)
    seeds = [11, 22, 33]

    frozen = frozen_pilot_quota(df, ["A"], level=10, seeds=seeds)

    per_seed = [compute_pilot_quota(df, ["A"], level=10, seed=seed) for seed in seeds]
    assert frozen == min(per_seed)
    # Feasible for every ordering: no selected patient can fall short of it.
    assert all(frozen <= value for value in per_seed)


def test_pilot_quota_respects_the_slide_contribution_cap() -> None:
    """A pilot cannot put 10% of its patches on one slide when the cap is 5%."""
    rows = []
    for patient in range(10):
        for slide, count in enumerate((9, 1)):
            rows.extend(
                {
                    "case_id": f"p{patient}",
                    "slide_id": f"p{patient}_s{slide}",
                    "cancer_type": "A",
                    "split": "train",
                }
                for _ in range(count)
            )
    frame = pd.DataFrame(rows)

    quota = compute_pilot_quota(frame, ["A"], level=10, seed=0)
    manifest = build_patch_pilot_manifest(frame, ["A"], 10, quota, seed=0)

    assert quota < 10
    assert manifest["slide_id"].value_counts().max() / len(manifest) <= 0.05


@pytest.mark.parametrize("seed", range(60))
def test_random_tail_assignment_is_distinct_from_native_and_rotated(seed: int) -> None:
    """The random permutation must not duplicate the native or rotated assignment."""
    assignments = build_tail_assignments(["A", "B", "C"], seed=seed, ordinal=False)

    orders = [tuple(order) for order in assignments.values()]
    assert len(set(orders)) == 3


def _write_seed_record(method_dir: Path, seed_idx: int) -> None:
    write_run_record(
        method_dir / f"seed={seed_idx}",
        {
            "benchmark": "patch",
            "class_names": ["A", "B"],
            "splits": {
                "test": {
                    "labels": [0, 1],
                    "preds": [0, 1],
                    "probabilities": [[0.9, 0.1], [0.2, 0.8]],
                    "logits": [[2.0, 0.0], [0.0, 2.0]],
                }
            },
        },
    )


def test_partial_confirmation_block_is_rejected(tmp_path: Path) -> None:
    """Fewer than five valid confirmation seeds must stop inference, not be averaged."""
    paths = {"results": tmp_path}
    method_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce"
    for seed_idx in range(3):  # only three of the required five present
        _write_seed_record(method_dir, seed_idx)

    with pytest.raises(RuntimeError, match="incomplete"):
        load_seed_predictions(paths, "severe", "weighted_ce", "native")


def test_complete_confirmation_block_stacks_all_five_seeds(tmp_path: Path) -> None:
    paths = {"results": tmp_path}
    method_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce"
    for seed_idx in range(5):
        _write_seed_record(method_dir, seed_idx)

    stacked = load_seed_predictions(paths, "severe", "weighted_ce", "native")

    assert stacked is not None
    assert stacked["preds"].shape[0] == 5


def test_missing_confirmation_method_is_not_silently_skipped(tmp_path: Path) -> None:
    """A roster method with no directory is a failed confirmation block."""
    with pytest.raises(RuntimeError, match="missing"):
        load_seed_predictions({"results": tmp_path}, "severe", "weighted_ce", "native")


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


def _ce_gate_entry(descriptive_only: bool) -> dict[str, object]:
    return {
        "method": "ce",
        "gate": "discrimination",
        "assignment": "native",
        "severity": "severe",
        "effect": 0.2,  # well above the 0.02 discrimination threshold
        "ci": (0.1, 0.3),  # excludes zero
        "descriptive_only": descriptive_only,
    }


def test_descriptive_only_cell_never_opens_a_gate_or_permutes() -> None:
    """A preflight descriptive-only cell must skip gates and permutation p-values."""
    from imbalance_benchmark.analysis.aggregate import _apply_gates

    def fake_p_value(entry, base_paths, config, seed):  # pragma: no cover - must not run
        raise AssertionError("descriptive-only cells must not be permutation tested")

    descriptive = [_ce_gate_entry(descriptive_only=True)]
    _apply_gates(descriptive, {}, {"dataset": {}}, 0, fake_p_value)
    assert descriptive[0]["gate_passed"] is False
    assert descriptive[0]["p_value"] is None

    confirmatory = [_ce_gate_entry(descriptive_only=False)]
    _apply_gates(confirmatory, {}, {"dataset": {}}, 0, lambda *_: 0.01)
    assert confirmatory[0]["gate_passed"] is True
    assert confirmatory[0]["p_value"] == 0.01


def test_tuning_selection_signed_lock_detects_tampering(tmp_path: Path) -> None:
    from imbalance_benchmark.common import sign_file, verify_signed_file, write_json

    selection = tmp_path / "tuning_selections.json"
    write_json(selection, {"native": {"severe": {"weighted_ce": {"lr": 1e-3}}}})
    sign_file(selection)

    verify_signed_file(selection)  # unaltered: passes

    write_json(selection, {"native": {"severe": {"weighted_ce": {"lr": 3e-3}}}})
    with pytest.raises(RuntimeError, match="no longer matches"):
        verify_signed_file(selection)

    unsigned = tmp_path / "tuning_selections_severe.json"
    write_json(unsigned, {})
    with pytest.raises(RuntimeError, match="no signed post-tuning lock"):
        verify_signed_file(unsigned)


def test_freeze_metadata_is_content_locked(tmp_path: Path) -> None:
    """Changing a frozen design field must be detected even without a CSV edit."""
    from imbalance_benchmark.common import write_json
    from imbalance_benchmark.manifest.freeze import lock_manifest_freeze, verify_manifest_freeze

    freeze_path = tmp_path / "manifest_freeze.json"
    write_json(freeze_path, {"shared_T": 100, "conditions": {}})
    freeze = lock_manifest_freeze({"shared_T": 100, "conditions": {}})
    verify_manifest_freeze(freeze)

    freeze["shared_T"] = 200
    with pytest.raises(RuntimeError, match="content"):
        verify_manifest_freeze(freeze)


def test_test_prediction_hash_is_prediction_sensitive() -> None:
    from imbalance_benchmark.modeling.workflows.confirmation import _test_prediction_hash

    base = {"test": {"labels": [0, 1], "preds": [0, 1], "probabilities": [[0.9, 0.1], [0.2, 0.8]]}}
    flipped = {"test": {"labels": [0, 1], "preds": [1, 0], "probabilities": [[0.9, 0.1], [0.2, 0.8]]}}

    assert _test_prediction_hash(base) == _test_prediction_hash(base)
    assert _test_prediction_hash(base) != _test_prediction_hash(flipped)


def test_clustered_endpoints_report_slide_and_patient_macro_nll_and_brier() -> None:
    from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
        clustered_endpoints,
    )

    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])
    predictions = probabilities.argmax(axis=1)
    identity = pd.DataFrame(
        {
            "case_id": ["p0", "p0", "p1", "p1"],
            "slide_id": ["s0", "s1", "s2", "s3"],
        }
    )

    out = clustered_endpoints(labels, predictions, probabilities, identity, seed=0)

    for key in (
        "slide_macro_balanced_accuracy",
        "patient_macro_balanced_accuracy",
        "slide_macro_f1",
        "patient_macro_f1",
        "slide_macro_nll",
        "patient_macro_nll",
        "slide_macro_brier",
        "patient_macro_brier",
    ):
        assert key in out and np.isfinite(out[key])


def test_weighted_ece_matches_scalar_ece_at_unit_weights() -> None:
    """The crossed-bootstrap ECE reduces to the scalar fixed-bin ECE at unit weights."""
    from imbalance_benchmark.analysis.inference.bootstrap import weighted_ece
    from imbalance_benchmark.analysis.metrics import expected_calibration_error

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=50)
    probs = rng.dirichlet(np.ones(3), size=50)
    row_weights = np.ones((50, 1), dtype=np.int64)

    weighted = weighted_ece(labels, probs, row_weights)[0]

    assert weighted == pytest.approx(expected_calibration_error(labels, probs))


def test_tail_recall_is_grouped_by_assignment(tmp_path: Path) -> None:
    """Tail recall must not be averaged across tail assignments and copied per row."""
    import sqlite3

    from imbalance_benchmark.analysis.db import init_schema
    from imbalance_benchmark.analysis.reporting.tables import _with_tail_recall

    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    # Same class 'C' is tail under assignment 'a' with recall 0.2 and under 'b'
    # with recall 0.8; grouping without assignment would report 0.5 for both.
    conn.executemany(
        "INSERT INTO runs (run_id, result_dir, benchmark, condition, assignment, method, seed_index) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            ("run_a", "d", "patch", "severe", "a", "weighted_ce", 0),
            ("run_b", "d", "patch", "severe", "b", "weighted_ce", 0),
        ],
    )
    conn.executemany(
        "INSERT INTO eval_classwise (run_id, split, class_name, tier, precision, recall, f1, support, nll, brier) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("run_a", "test", "C", "tail", 0.0, 0.2, 0.0, 5, 0.0, 0.0),
            ("run_b", "test", "C", "tail", 0.0, 0.8, 0.0, 5, 0.0, 0.0),
        ],
    )
    conn.commit()

    summary = pd.DataFrame(
        {
            "assignment": ["a", "b"],
            "condition": ["severe", "severe"],
            "method": ["weighted_ce", "weighted_ce"],
        }
    )
    merged = _with_tail_recall(summary, conn, "test").set_index("assignment")

    assert merged.loc["a", "tail_recall"] == pytest.approx(0.2)
    assert merged.loc["b", "tail_recall"] == pytest.approx(0.8)


def test_recovery_standard_error_uses_the_recovery_distribution() -> None:
    """recovery_se must come from numerator/denominator, not the raw effect spread."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import (
        _recovery_standard_error,
        _standard_error,
    )

    rng = np.random.default_rng(1)
    numerator = rng.normal(0.3, 0.05, size=500)
    denominator = rng.normal(0.5, 0.05, size=500)
    comparison = {
        "bootstrap_effect": numerator.tolist(),
        "bootstrap_numerator": numerator.tolist(),
        "bootstrap_denominator": denominator.tolist(),
    }

    recovery_se = _recovery_standard_error(comparison)
    effect_se = _standard_error(comparison)

    expected = float(np.nanstd(numerator / denominator, ddof=1))
    assert recovery_se == pytest.approx(expected)
    assert recovery_se != pytest.approx(effect_se)


def _rq3_cell(group: str, method: str, rho: float, deficit: float, gate: bool) -> dict:
    return {
        "group": group,
        "method": method,
        "rho": rho,
        "separability": 0.5,
        "learnability": 0.4,
        "log_min_support": 3.0,
        "is_wsi": 0.0 if "patch" in group else 1.0,
        "gate_passed": gate,
        "deficit_ba": deficit,
        "deficit_se": 0.01,
        "recovery": 0.5,
        "recovery_se": 0.1,
    }


def test_cross_dataset_rq3_pools_groups_and_reports_stability() -> None:
    """RQ3's combined fit spans dataset-target groups with LODO and sensitivity fits."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import cross_dataset_rq3

    cells = []
    for i, group in enumerate(["tcga:patch", "tcga:wsi", "bracs:patch", "panda:wsi"]):
        cells.append(_rq3_cell(group, "ce", 10.0 + i, 0.05 + 0.01 * i, gate=True))
        cells.append(_rq3_cell(group, "weighted_ce", 10.0 + i, np.nan, gate=True))

    report = cross_dataset_rq3(cells)

    assert report["n_groups"] == 4
    assert len(report["models"]["deficit"]["rand_intercepts"]) == 4
    assert set(report["sensitivity"]) == {"learnability", "log_min_support", "is_wsi"}
    assert set(report["leave_one_group_out"]) == set(report["groups"])


def test_rq3_equal_averages_split_repetitions_by_dataset_target(tmp_path: Path) -> None:
    """Three patient splits are fixed repetitions, not independent RQ3 cells."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import load_rq3_cells
    from imbalance_benchmark.common import write_json

    for split, deficit in enumerate((0.1, 0.2, 0.3)):
        write_json(
            tmp_path / f"split={split}" / "data" / "rq3.json",
            {
                "cells": [
                    {
                        "group": "tcga-ut",
                        "assignment": "native",
                        "severity": "severe",
                        "method": "ce",
                        "rho": 10.0,
                        "separability": 0.5,
                        "learnability": 0.4,
                        "log_min_support": 2.0,
                        "log_effective_support": 1.0,
                        "is_wsi": 0.0,
                        "gate_passed": True,
                        "deficit_ba": deficit,
                        "deficit_se": 0.01,
                        "recovery": np.nan,
                        "recovery_se": np.nan,
                    }
                ]
            },
        )
    write_json(
        tmp_path / "data" / "cross_split_gates_and_recovery.json",
        {
            "comparisons": [
                {
                    "assignment": "native",
                    "severity": "severe",
                    "method": "ce",
                    "gate": "discrimination",
                    "gate_passed": True,
                    "bootstrap_effect": [0.1, 0.2],
                },
                {
                    "assignment": "native",
                    "severity": "severe",
                    "method": "ce",
                    "gate": "discrimination",
                    "gate_passed": True,
                    "bootstrap_effect": [0.1, 0.2],
                },
                {
                    "assignment": "native",
                    "severity": "severe",
                    "method": "ce",
                    "gate": "discrimination",
                    "gate_passed": True,
                    "bootstrap_effect": [0.1, 0.2, 0.3],
                }
            ]
        },
    )

    cells = load_rq3_cells([tmp_path])

    assert len(cells) == 1
    assert cells[0]["group"] == "tcga-ut"
    assert cells[0]["deficit_ba"] == pytest.approx(0.2)


def test_rq3_cells_keep_assignment_and_severity_and_dataset_target_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """RQ3 cells retain their crossed identity and never merge a dataset's targets."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import _cells, run_rq3

    observed_regimes: list[bool] = []

    def covariates(_: dict, is_mil: bool, __: dict, *args: object) -> dict[str, float]:
        observed_regimes.append(is_mil)
        return {
            "separability": 0.5,
            "learnability": 0.4,
            "log_min_support": 2.0,
            "log_effective_support": 1.0,
            "is_wsi": 1.0,
        }

    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._covariates",
        covariates,
    )
    comparisons = [
        {
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "discrimination",
            "gate_passed": True,
            "effect": 0.1,
            "bootstrap_effect": [0.05, 0.15],
        }
    ]
    freeze = {
        "assignment_conditions": {
            "native": {"severe": {"achieved_rho": 10.0, "contribution_stats": {}, "path": str(tmp_path / "x.csv")}}
        }
    }

    cells = _cells({"data": tmp_path}, comparisons, freeze, "panda:wsi", True)
    report = run_rq3(
        {"data": tmp_path},
        {"dataset": {"name": "panda", "regime": "patch", "target": "changed_target"}},
        {
            **freeze,
            "dataset_provenance": {
                "name": "panda",
                "regime": "wsi",
                "target": "isup_grade",
            },
        },
        comparisons,
    )

    assert cells[0]["assignment"] == "native"
    assert cells[0]["severity"] == "severe"
    assert report["cells"][0]["group"] == "panda:isup_grade"
    assert observed_regimes[-1] is True


def test_rq3_cross_split_values_come_from_crossed_bootstrap(tmp_path: Path) -> None:
    """RQ3 uses the equal-split gate and ratio distribution, not split-level averages."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import load_rq3_cells
    from imbalance_benchmark.common import write_json

    cell = {
        "group": "tcga-ut:patch",
        "assignment": "native",
        "severity": "severe",
        "method": "weighted_ce",
        "rho": 10.0,
        "separability": 0.5,
        "learnability": 0.4,
        "log_min_support": 2.0,
        "log_effective_support": 1.0,
        "is_wsi": 0.0,
        "gate_passed": False,
        "deficit_ba": np.nan,
        "deficit_se": np.nan,
        "recovery": 0.2,
        "recovery_se": 0.01,
    }
    for split in range(3):
        write_json(tmp_path / f"split={split}" / "data" / "rq3.json", {"cells": [cell]})
    write_json(
        tmp_path / "data" / "cross_split_gates_and_recovery.json",
        {
            "comparisons": [
                {
                    "assignment": "native",
                    "severity": "severe",
                    "method": "ce",
                    "gate": "discrimination",
                    "gate_passed": True,
                    "bootstrap_effect": [0.1, 0.2],
                },
                {
                    "assignment": "native",
                    "severity": "severe",
                    "method": "weighted_ce",
                    "gate": "discrimination",
                    "gate_passed": True,
                    "bootstrap_numerator": [1.0, 4.0],
                    "bootstrap_denominator": [2.0, 2.0],
                }
            ]
        },
    )

    cells = load_rq3_cells([tmp_path])

    assert cells[0]["gate_passed"] is True
    assert cells[0]["recovery"] == pytest.approx(1.25)
    assert cells[0]["recovery_se"] == pytest.approx(np.std([0.5, 2.0], ddof=1))


def test_balanced_predictions_use_one_unassigned_result_directory(tmp_path: Path) -> None:
    """Assignment-specific analyses reuse one balanced record rather than copies."""
    paths = {"results": tmp_path / "results"}
    balanced = paths["results"] / "assignment=unassigned" / "balanced" / "ce"
    balanced.mkdir(parents=True)

    resolved = _confirmation_dir(paths, "balanced", "ce", "reversed")

    assert resolved == balanced
    assert not (paths["results"] / "assignment=reversed" / "balanced").exists()


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


def test_mil_covariates_use_the_dataset_slide_identity_not_raw_chunk_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A slide represented by feature chunks still contributes one MIL identity row."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import _covariates

    manifest = tmp_path / "manifest.csv"
    balanced = tmp_path / "manifest_balanced.csv"
    condition = tmp_path / "condition.csv"
    rows = [
        {"case_id": "p0", "slide_id": "s0", "cancer_type": "A", "feature_path": "a.pt"},
        {"case_id": "p0", "slide_id": "s0", "cancer_type": "A", "feature_path": "b.pt"},
        {"case_id": "p1", "slide_id": "s1", "cancer_type": "B", "feature_path": "c.pt"},
    ]
    pd.DataFrame(rows).to_csv(manifest, index=False)
    pd.DataFrame(rows).to_csv(balanced, index=False)
    pd.DataFrame(rows).to_csv(condition, index=False)

    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._feature_frame",
        lambda *_: (features, labels),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.intrinsic_separability",
        lambda *_: {"linear_probe_macro_recall": 0.5, "knn_macro_recall": 0.5, "per_class_nn_error": {}},
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.condition_learnability",
        lambda *_: {"linear_probe_macro_recall": 0.5},
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.class_margin_cross_fit",
        lambda *_: np.array([0.1, 0.2]),
    )

    result = _covariates(
        {"data": tmp_path}, True, {"path": str(condition), "contribution_stats": {}}
    )

    assert np.isfinite(result["log_effective_support"])


def test_freeze_verifies_pilot_and_prepared_manifest_artifacts(tmp_path: Path) -> None:
    """Held-out manifest or pilot changes invalidate the frozen record."""
    from imbalance_benchmark.common import compute_sha256, sign_file, write_json
    from imbalance_benchmark.manifest.freeze import verify_manifest_freeze

    pilot = tmp_path / "pilot_report.json"
    manifest = tmp_path / "manifest.csv"
    write_json(pilot, {"definitive_floor": 10})
    write_json(manifest, {"held_out": "locked"})
    sign_file(pilot)
    meta = {
        "content_sha256": "",
        "pilot_report": {"path": str(pilot), "sha256": compute_sha256(pilot)},
        "prepared_manifest": {"path": str(manifest), "sha256": compute_sha256(manifest)},
    }
    from imbalance_benchmark.manifest.freeze import lock_manifest_freeze

    frozen = lock_manifest_freeze(meta)
    verify_manifest_freeze(frozen)
    write_json(manifest, {"held_out": "changed"})
    with pytest.raises(RuntimeError, match="Prepared manifest altered"):
        verify_manifest_freeze(frozen)
