from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.commands.pilot import _pilot_report_payload
from imbalance_benchmark.manifest.pilot.candidates import (
    build_patch_pilot_manifest,
    compute_pilot_quota,
    frozen_pilot_quota,
    meets_method_floor,
)
from imbalance_benchmark.manifest.pilot.training import fit_pilot_model
from typing import Any

def test_pilot_training_uses_complete_bags_and_the_run_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The WSI pilot loads whole bags and trains under the run's own config."""
    observed: dict[str, Any] = {}

    class Dataset:
        def __init__(self, *_: object, **kwargs: object) -> None:
            observed["bag_kwargs"] = kwargs

        def get_int_targets(self) -> torch.Tensor:
            return torch.tensor([0, 1])

        def __len__(self) -> int:
            return 2

    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot.training.BagFeatureDataset", Dataset
    )

    def fit_model(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
        observed["context"] = ctx
        return {}, 0.5

    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot.training.fit_model", fit_model
    )

    fit_pilot_model(
        tmp_path / "pilot.csv",
        torch.device("cpu"),
        2,
        True,
        object(),
        initialization_seed=3,
        config={"wsi_training": {"bag_batch_size": 2}},
    )

    assert observed["bag_kwargs"] == {"device": torch.device("cpu")}
    assert observed["context"]["config"]["wsi_training"]["bag_batch_size"] == 2

def test_pilot_definitive_floor_does_not_collapse_patient_and_slide_floors() -> None:
    """Patch pilot levels count patients; the slide floor must not become a 20-patient floor."""
    levels = [5, 10, 15, 20, 30]
    flat_ba = {seed: [0.5] * len(levels) for seed in (0, 1, 2)}
    flat_recall = {seed: [[0.5, 0.5]] * len(levels) for seed in (0, 1, 2)}
    support = {"A": {"patients": 12, "slides": 25}, "B": {"patients": 12, "slides": 25}}

    difficulty = {
        seed: {
            "linear_probe_recall": {"A": 0.7, "B": 0.5},
            "knn_recall": {"A": 0.6, "B": 0.4},
        }
        for seed in (0, 1, 2)
    }
    patch = _pilot_report_payload(
        levels, levels, False, False, [0, 1, 2], {}, flat_ba, flat_recall, support, difficulty
    )
    mil = _pilot_report_payload(
        levels, levels, True, False, [0, 1, 2], {}, flat_ba, flat_recall, support, difficulty
    )

    # Patch pilot counts patients -> patient floor 10, not the 20-slide floor.
    assert patch["stability_floor"] == 5
    assert patch["definitive_floor"] == 10
    assert patch["excluded"] is False
    assert patch["dropped_levels"] == []
    # MIL pilot counts slides -> slide floor 20 applies to the level dimension.
    assert mil["definitive_floor"] == 20
    assert mil["dropped_levels"] == []
    assert patch["difficulty_evidence"]["ranking_easiest_to_hardest"] == ["A", "B"]

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

    constraints = _pilot_constraints(report)

    assert constraints.independent_floor == 30


def test_patch_floor_does_not_track_the_pilot_quota(tmp_path: Path) -> None:
    """Two splits differing only in pilot quota must freeze the same patch floor.

    The quota is pinned by the scarcest class's least-stocked patient at the
    largest pilot level, so letting it scale the floor makes one one-patch
    patient decide a split's achievable severity.
    """
    from imbalance_benchmark.manifest.freezing import _pilot_constraints

    floors = []
    for quota in (1, 24):
        report = tmp_path / f"pilot_report_{quota}.json"
        report.write_text(
            json.dumps(
                {"definitive_floor": 20, "quotas": {"1": quota}, "excluded": False}
            ),
            encoding="utf-8",
        )
        floors.append(_pilot_constraints(report).patch_floor)

    assert floors[0] == floors[1] == 20

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

    frozen, feasible_levels = frozen_pilot_quota(df, ["A"], levels=[10], seeds=seeds)

    per_seed = [compute_pilot_quota(df, ["A"], level=10, seed=seed) for seed in seeds]
    assert frozen == min(per_seed)
    # Feasible for every ordering: no selected patient can fall short of it.
    assert all(frozen <= value for value in per_seed)
    assert feasible_levels == [10]

def test_pilot_quota_ignores_a_single_low_inventory_patient() -> None:
    """One patient with a single patch must not drag the whole class's quota to 1.

    The candidate quota is the level-th order statistic of per-patient
    inventory, and eligibility for it is inventory-based, so a scarce patient
    is excluded from that level's pilot rather than pinning the quota for
    everyone else.
    """
    rows = []
    for patient in range(11):
        for slide in range(2):
            rows.extend(
                {
                    "case_id": f"p{patient}",
                    "slide_id": f"p{patient}_s{slide}",
                    "cancer_type": "A",
                    "split": "train",
                }
                for _ in range(15)
            )
    rows.append(
        {
            "case_id": "p_poisoned",
            "slide_id": "p_poisoned_s0",
            "cancer_type": "A",
            "split": "train",
        }
    )
    frame = pd.DataFrame(rows)

    quota = compute_pilot_quota(frame, ["A"], level=10, seed=0)

    assert quota == 30

def test_patch_pilot_manifest_excludes_a_patient_too_scarce_for_the_frozen_quota() -> (
    None
):
    """Building the manifest at the frozen quota must skip an ineligible patient.

    ``compute_pilot_quota`` already ignores a low-inventory patient when
    choosing the quota; ``build_patch_pilot_manifest`` must apply the same
    inventory-based eligibility filter when it later selects patients at
    that frozen quota, or the poisoned patient still lands in the level-10
    prefix and cannot supply the required 30 patches.
    """
    rows = []
    for patient in range(11):
        for slide in range(2):
            rows.extend(
                {
                    "case_id": f"p{patient}",
                    "slide_id": f"p{patient}_s{slide}",
                    "cancer_type": "A",
                    "split": "train",
                }
                for _ in range(15)
            )
    rows.append(
        {
            "case_id": "p_poisoned",
            "slide_id": "p_poisoned_s0",
            "cancer_type": "A",
            "split": "train",
        }
    )
    frame = pd.DataFrame(rows)
    quota = compute_pilot_quota(frame, ["A"], level=10, seed=0)

    manifest = build_patch_pilot_manifest(frame, ["A"], level=10, quota=quota, seed=0)

    assert "p_poisoned" not in set(manifest["case_id"])
    assert manifest["case_id"].nunique() == 10
    assert len(manifest) == 10 * quota

def test_frozen_pilot_quota_drops_levels_the_cap_cannot_satisfy(monkeypatch) -> None:
    """The contribution cap tightens as the level shrinks, so a level no quota
    can satisfy is dropped rather than silently reused from a larger level."""
    from imbalance_benchmark.manifest.pilot import training as pilot_training

    def fake_compute(df, classes, level, seed):
        if level == 10:
            raise ValueError("Pilot inventory cannot satisfy patient and slide caps")
        return {5: 40, 15: 20, 20: 30}[level]

    monkeypatch.setattr(pilot_training, "compute_pilot_quota", fake_compute)

    quota, feasible_levels = pilot_training.frozen_pilot_quota(
        pd.DataFrame(), ["A"], levels=[5, 10, 15, 20], seeds=[0]
    )

    # Level 10 is infeasible at any quota, so it's dropped, not silently reused.
    assert feasible_levels == [5, 15, 20]
    assert quota == 20

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

def test_method_floor_requires_patients_and_slides_together() -> None:
    assert not meets_method_floor(
        {"patients": 9, "slides": 100}, patient_equals_slide=False
    )
    assert not meets_method_floor(
        {"patients": 100, "slides": 19}, patient_equals_slide=False
    )
    assert meets_method_floor(
        {"patients": 10, "slides": 20}, patient_equals_slide=False
    )


def test_grouped_difficulty_uses_case_disjoint_folds() -> None:
    from imbalance_benchmark.manifest.pilot.difficulty import grouped_difficulty

    features, labels, groups = [], [], []
    for label in range(2):
        for case in range(5):
            for row in range(2):
                features.append([label * 10 + case, row])
                labels.append(label)
                groups.append(f"{label}_{case}")
    evidence = grouped_difficulty(
        np.asarray(features), np.asarray(labels), np.asarray(groups), ["A", "B"]
    )

    assert len(evidence["folds"]) == 5
    assert all(
        not set(fold["train_groups"]) & set(fold["test_groups"])
        for fold in evidence["folds"]
    )
