from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from imbalance_benchmark.analysis.inference.permutation import (
    paired_block_permutation_ba,
)
from imbalance_benchmark.commands.freeze import cmd_freeze
from imbalance_benchmark.commands.prepare import cmd_prepare
from imbalance_benchmark.common import sign_file
from imbalance_benchmark.construction import allocate_counts, max_shared_total
from imbalance_benchmark.construction import locked_class_names
from imbalance_benchmark.manifest.shared_total import search as shared_total_search
from imbalance_benchmark.manifest.shared_total.spreading import SPREAD_ASSIGNMENTS_BY_DATASET
from imbalance_benchmark.datasets.data import BagFeatureDataset
from imbalance_benchmark.manifest.construction_helpers import _retains_fixed_pool
from imbalance_benchmark.manifest.shared_total.search import cap_feasible_shared_total
from imbalance_benchmark.manifest.freezing import _build_conditions
from imbalance_benchmark.manifest.pilot.candidates import PilotFit, run_pilot_seed
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.modeling.workflows.confirmation import RunContext, confirm_ce

def test_pilot_construction_and_initialization_seeds_are_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[int] = []

    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot.candidates.build_patch_pilot_manifest",
        lambda *_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot.candidates.evaluate_pilot_candidate",
        lambda _df, _scratch, fit: (
            observed.append(fit.initialization_seed) or (0.5, [0.5])
        ),
    )

    construction_seed = 17
    run_pilot_seed(
        pd.DataFrame(),
        ["A"],
        [5],
        construction_seed,
        tmp_path,
        1,
        PilotFit(object(), torch.device("cpu"), 1, False, 101),
    )

    assert observed == [101]

def test_native_tail_order_uses_bracs_clinical_label_order() -> None:
    rows = [
        {"cancer_type": label, "split": split}
        for split in ("train", "validation", "test")
        for label in ("IC", "N", "ADH", "PB", "DCIS", "UDH", "FEA")
    ]

    assert locked_class_names(pd.DataFrame(rows)) == [
        "N",
        "PB",
        "UDH",
        "FEA",
        "ADH",
        "DCIS",
        "IC",
    ]

def test_shared_total_keeps_all_naturally_balanced_support() -> None:
    available = [100, 100, 100]
    shared_total = max_shared_total(available, min_support=10)

    assert shared_total == sum(available)
    for ratio in (1.0, 10.0, 100.0):
        allocation = allocate_counts(available, shared_total, ratio, min_support=10)
        assert sum(allocation) == shared_total
        assert all(count <= support for count, support in zip(allocation, available))

def test_shared_total_search_fails_when_moderate_and_severe_maxima_never_coincide() -> (
    None
):
    """Moderate and severe can each cap at the same ratio without ever doing so together.

    Two equally-sized classes: the achieved-moderate maximum and the
    achieved-severe maximum are each real, individually reachable values,
    but no single total realizes both at once. Summing the two achieved
    ratios (the old approach) would silently pick a compromise total that
    matches neither window; the report's protocol instead requires a single
    total to realize both maxima at once, so the exact search must refuse
    rather than substitute a compromise. See
    ``test_shared_total_search.test_no_simultaneous_maxima_raises_rather_than_a_summed_compromise``
    for the same property with the maxima windows inspected directly.
    """
    frame = pd.DataFrame(
        [
            {
                "case_id": f"{name}_{index}",
                "slide_id": f"{name}_{index}",
                "cancer_type": name,
            }
            for name, support in (("A", 400), ("B", 400))
            for index in range(support)
        ]
    )

    with pytest.raises(ValueError, match="simultaneously"):
        cap_feasible_shared_total(
            frame,
            ["A", "B"],
            min_support=10,
            is_mil=True,
            seed=1,
            independent_floor=10,
        )

def test_retains_fixed_pool_checks_full_designated_coverage() -> None:
    """The primitive itself is unchanged: it still tests pool-vs-selection coverage.

    What changed is *where* it is enforced - only the largest count a class
    receives must satisfy it; see
    ``test_pool_availability_lets_smaller_conditions_leave_units_unused``.
    """
    pool = pd.DataFrame({"case_id": ["p1", "p2", "p3"], "slide_id": ["s1", "s2", "s3"]})
    smaller_condition = pool.iloc[:2]

    assert not _retains_fixed_pool(smaller_condition, pool)
    assert _retains_fixed_pool(pool, pool)
    assert not _retains_fixed_pool(
        pd.DataFrame({"case_id": ["p9"], "slide_id": ["s9"]}), pool
    )

def test_pool_availability_lets_smaller_conditions_leave_units_unused(
    tmp_path: Path,
) -> None:
    """Severity-skewed availability must still yield a controlled ratio > 1.

    A scarce tail class's severe/moderate count can sit near the support
    floor while its balanced count is far larger. The fixed pool is sized to
    the largest count a class ever receives (here, balanced); the pool
    availability invariant only requires *that* condition to retain every
    designated unit, not the smaller ones too. Requiring every condition to
    retain the same pool (the old, strict semantics) makes a scarce class's
    pool contradictorily small for its own larger balanced count, which is
    what collapsed BRACS patch splits 0/1 to an unachieved ratio of 1.
    """
    rows = [
        {
            "case_id": f"A-patient-{patient}",
            "slide_id": f"A-slide-{patient}-{slide}",
            "patch_id": f"A-{patient}-{slide}-{patch}",
            "cancer_type": "A",
            "split": "train",
        }
        for patient in range(40)
        for slide in range(4)
        for patch in range(5)
    ] + [
        {
            "case_id": f"B-patient-{patient}",
            "slide_id": f"B-slide-{patient}-{slide}",
            "patch_id": f"B-{patient}-{slide}-{patch}",
            "cancer_type": "B",
            "split": "train",
        }
        for patient in range(15)
        for slide in range(2)
        for patch in range(3)
    ]
    frame = pd.DataFrame(rows)

    total = cap_feasible_shared_total(frame, ["A", "B"], 20, False, 1)
    conditions = _build_conditions(
        frame, ["A", "B"], total, 20, False, 1, tmp_path, independent_floor=10
    )

    assert conditions["moderate"]["achieved_rho"] > 1.5
    assert conditions["severe"]["achieved_rho"] >= conditions["moderate"]["achieved_rho"]
    assert (
        conditions["severe"]["allocated_counts"]["B"]
        < conditions["balanced"]["allocated_counts"]["B"]
    )

    retained = {
        name: pd.read_csv(condition["path"])
        .groupby("cancer_type")[["case_id", "slide_id"]]
        .agg(lambda values: frozenset(values))
        for name, condition in conditions.items()
    }
    largest = max(conditions, key=lambda name: conditions[name]["allocated_counts"]["B"])
    for name in conditions:
        assert retained[name].loc["B", "case_id"] <= retained[largest].loc["B", "case_id"]
        assert retained[name].loc["B", "slide_id"] <= retained[largest].loc["B", "slide_id"]

def test_feasible_patch_total_builds_with_one_retained_pool(tmp_path: Path) -> None:
    rows = [
        {
            "case_id": f"{class_name}-patient-{patient}",
            "slide_id": f"{class_name}-slide-{patient}-{slide}",
            "patch_id": f"{class_name}-{patient}-{slide}-{patch}",
            "cancer_type": class_name,
            "split": "train",
        }
        for class_name, patients in (("A", 15), ("B", 11))
        for patient in range(patients)
        for slide in range(4 if class_name == "A" or patient < 7 else 3)
        for patch in range(2)
    ]
    frame = pd.DataFrame(rows)
    total = cap_feasible_shared_total(frame, ["A", "B"], 20, False, 1)

    conditions = _build_conditions(
        frame, ["A", "B"], total, 20, False, 1, tmp_path, independent_floor=10
    )

    retained_units = {
        name: pd.read_csv(condition["path"])
        .groupby("cancer_type")[["case_id", "slide_id"]]
        .agg(lambda values: frozenset(values))
        for name, condition in conditions.items()
    }
    # Each condition draws from the class's fixed pool; only the condition with
    # the largest count for a class must retain every designated unit in it.
    for cls in ("A", "B"):
        largest = max(
            conditions, key=lambda name: conditions[name]["allocated_counts"][cls]
        )
        for name in conditions:
            assert (
                retained_units[name].loc[cls, "case_id"]
                <= retained_units[largest].loc[cls, "case_id"]
            )
            assert (
                retained_units[name].loc[cls, "slide_id"]
                <= retained_units[largest].loc[cls, "slide_id"]
            )

def test_shared_total_search_handles_non_monotone_contribution_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "case_id": f"{class_name}-patient-{patient}",
            "slide_id": f"{class_name}-slide-{patient}-{slide}",
            "cancer_type": class_name,
        }
        for class_name in ("A", "B")
        for patient in range(11)
        for slide in range(5 if patient < 5 else 4)
    ]

    real_probe = shared_total_search._cap_feasible
    probes = []

    def tracked_probe(*args):
        probes.append(1)
        return real_probe(*args)

    monkeypatch.setattr(shared_total_search, "_cap_feasible", tracked_probe)
    total = cap_feasible_shared_total(
        pd.DataFrame(rows), ["A", "B"], min_support=20, is_mil=True, seed=1
    )

    assert total == 64
    assert len(probes) < 10

def test_confirmation_training_context_receives_the_validation_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[object] = []
    val_loader = object()
    run = RunContext(
        torch.device("cpu"),
        {},
        2,
        False,
        val_loader,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        {"results": tmp_path},
        [7],
        ["A", "B"],
        "native",
    )

    def fake_context(*args: object) -> dict[str, object]:
        seen.append(args[-1])
        return {
            "model": torch.nn.Linear(1, 1),
            "train_dataset": [0],
            "seed": 7,
            "param_config": {},
        }

    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation.build_training_ctx",
        fake_context,
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._timed_fit",
        lambda _fit, _ctx: ({}, 0.0),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._run_and_record",
        lambda *args: None,
    )

    confirm_ce("balanced", {"lr": 1e-3}, object(), run)  # type: ignore[arg-type]

    assert seen == [val_loader]

def test_prepare_writes_three_distinct_patient_split_manifests(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"paths": {"outputs": str(tmp_path / "outputs")}}, handle)

    cmd_prepare(Namespace(config=str(config_path), seed=3, split_index=None))

    manifests = [
        pd.read_csv(tmp_path / "outputs" / f"split={index}" / "data" / "manifest.csv")
        for index in range(3)
    ]
    assert all(
        set(frame["split"]) == {"train", "validation", "test"} for frame in manifests
    )
    assert any(
        not manifests[0][["case_id", "split"]].equals(frame[["case_id", "split"]])
        for frame in manifests[1:]
    )

def test_prepare_excludes_configured_classes_before_splitting(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "paths": {"outputs": str(tmp_path / "outputs")},
                "dataset": {"excluded_classes": ["class_A"]},
            },
            handle,
        )

    cmd_prepare(Namespace(config=str(config_path), seed=3, split_index=None))

    manifest = pd.read_csv(tmp_path / "outputs" / "split=0" / "data" / "manifest.csv")

    assert "class_A" not in set(manifest["cancer_type"])
    assert {"class_B", "class_C", "class_D"}.issubset(set(manifest["cancer_type"]))

def test_freeze_uses_the_resampling_seed_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resampling seed provenance does not require the unrelated spread arm."""
    monkeypatch.setitem(SPREAD_ASSIGNMENTS_BY_DATASET, "synthetic", ())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"outputs": str(tmp_path / "outputs")},
                "dataset": {
                    "name": "synthetic",
                    "regime": "patch",
                    "target": "synthetic_target",
                    "version": "test-fixture-v1",
                    "eligibility_rules": {"fixture": True},
                },
                "analysis": {"bootstrap_replicates": 2},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "case_id": f"{cls}_{index}",
            "slide_id": f"{cls}_{index}",
            "patch_id": f"{cls}_{index}_patch",
            "cancer_type": cls,
            "split": "train" if index < train_n else "test",
        }
        # Three classes, each comfortably above the 20-patch floor whichever
        # role a tail assignment gives it, so moderate/severe can genuinely
        # differ instead of both saturating the same head-capacity ceiling.
        # C's train count (100) is chosen so the moderate and severe rho
        # maxima are simultaneously attainable across native and reversed
        # assignments - the exact search refuses a total otherwise.
        for cls, total, train_n in (("A", 340, 300), ("B", 240, 200), ("C", 140, 100))
        for index in range(total)
    ]
    for split_index in range(3):
        data_dir = tmp_path / "outputs" / f"split={split_index}" / "data"
        data_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(data_dir / "manifest.csv", index=False)
        (data_dir / "pilot_report.json").write_text(
            json.dumps(
                {
                    "definitive_floor": 10,
                    "quotas": {"0": 1},
                    "excluded": False,
                    "difficulty_evidence": {
                        "difficulty": {"A": 0.1, "B": 0.2, "C": 0.3}
                    },
                }
            ),
            encoding="utf-8",
        )
        sign_file(data_dir / "pilot_report.json")

    cmd_freeze(Namespace(config=str(config_path), seed=7, split_index=0))

    freeze = json.loads(
        (tmp_path / "outputs" / "split=0" / "data" / "manifest_freeze.json").read_text()
    )
    preflight = json.loads(Path(freeze["bootstrap_preflight"]["path"]).read_text())
    assert preflight["seed"] == derive_seed(7, "resampling")

def test_two_seed_permutation_stack_is_not_mistaken_for_one_probability_matrix():
    labels = np.array([0, 1, 0, 1])
    case_ids = np.array(["P0", "P1", "P2", "P3"])
    method = np.array([[0, 1, 0, 1], [0, 1, 1, 1]])
    ce = np.array([[1, 1, 0, 1], [1, 0, 1, 1]])
    p_value = paired_block_permutation_ba(labels, method, ce, case_ids, n_classes=2)
    assert 0.0 <= p_value <= 1.0
