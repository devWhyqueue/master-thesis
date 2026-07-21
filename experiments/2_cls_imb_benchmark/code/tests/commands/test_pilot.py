from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from imbalance_benchmark.commands.pilot import _pilot_report_payload
from imbalance_benchmark.manifest.pilot import (
    build_patch_pilot_manifest,
    compute_pilot_quota,
    frozen_pilot_quota,
    meets_method_floor,
)
from imbalance_benchmark.manifest.pilot_training import fit_pilot_model
from typing import Any

def test_pilot_training_receives_the_configured_wsi_evidence_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The WSI pilot must use the same fixed instance cap and training config."""
    observed: dict[str, Any] = {}

    class Dataset:
        def __init__(self, *_: object, **kwargs: object) -> None:
            observed["bag_kwargs"] = kwargs

        def get_int_targets(self) -> torch.Tensor:
            return torch.tensor([0, 1])

        def __len__(self) -> int:
            return 2

    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot_training.BagFeatureDataset", Dataset
    )

    def fit_model(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
        observed["context"] = ctx
        return {}, 0.5

    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot_training.fit_model", fit_model
    )

    fit_pilot_model(
        tmp_path / "pilot.csv",
        torch.device("cpu"),
        2,
        True,
        object(),
        initialization_seed=3,
        config={"wsi_training": {"max_instances": 17, "bag_batch_size": 2}},
        bag_kwargs={"max_instances": 17, "instance_selection_seed": 11},
    )

    assert observed["bag_kwargs"] == {
        "device": torch.device("cpu"),
        "max_instances": 17,
        "instance_selection_seed": 11,
    }
    assert observed["context"]["config"]["wsi_training"]["max_instances"] == 17

def test_pilot_definitive_floor_does_not_collapse_patient_and_slide_floors() -> None:
    """Patch pilot levels count patients; the slide floor must not become a 20-patient floor."""
    levels = [5, 10, 15, 20, 30]
    flat_ba = {seed: [0.5] * len(levels) for seed in (0, 1, 2)}
    flat_recall = {seed: [[0.5, 0.5]] * len(levels) for seed in (0, 1, 2)}
    support = {"A": {"patients": 12, "slides": 25}, "B": {"patients": 12, "slides": 25}}

    patch = _pilot_report_payload(
        levels, levels, False, False, [0, 1, 2], {}, flat_ba, flat_recall, support
    )
    mil = _pilot_report_payload(
        levels, levels, True, False, [0, 1, 2], {}, flat_ba, flat_recall, support
    )

    # Patch pilot counts patients -> patient floor 10, not the 20-slide floor.
    assert patch["stability_floor"] == 5
    assert patch["definitive_floor"] == 10
    assert patch["excluded"] is False
    assert patch["dropped_levels"] == []
    # MIL pilot counts slides -> slide floor 20 applies to the level dimension.
    assert mil["definitive_floor"] == 20
    assert mil["dropped_levels"] == []

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

    frozen, feasible_levels = frozen_pilot_quota(df, ["A"], levels=[10], seeds=seeds)

    per_seed = [compute_pilot_quota(df, ["A"], level=10, seed=seed) for seed in seeds]
    assert frozen == min(per_seed)
    # Feasible for every ordering: no selected patient can fall short of it.
    assert all(frozen <= value for value in per_seed)
    assert feasible_levels == [10]

def test_frozen_pilot_quota_drops_levels_the_cap_cannot_satisfy(monkeypatch) -> None:
    """The contribution cap tightens as the level shrinks, so a level no quota
    can satisfy is dropped rather than silently reused from a larger level."""
    from imbalance_benchmark.manifest import pilot_training

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
