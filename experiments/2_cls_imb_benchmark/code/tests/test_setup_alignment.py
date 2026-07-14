from __future__ import annotations

from pathlib import Path
from argparse import Namespace
import json

import pandas as pd
import numpy as np
import pytest
import torch
import yaml

from imbalance_benchmark.analysis.inference.permutation import paired_block_permutation_ba
from imbalance_benchmark.modeling.workflows.confirmation import RunContext, confirm_ce
from imbalance_benchmark.commands.freeze import cmd_freeze
from imbalance_benchmark.commands.prepare import cmd_prepare
from imbalance_benchmark.construction import allocate_counts, max_shared_total
from imbalance_benchmark.manifest.construction_helpers import cap_feasible_shared_total
from imbalance_benchmark.manifest.freeze import achieved_rho
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.common import sign_file
from imbalance_benchmark.datasets.data import BagFeatureDataset


def test_shared_total_keeps_all_naturally_balanced_support() -> None:
    available = [100, 100, 100]
    shared_total = max_shared_total(available, min_support=10)

    assert shared_total == sum(available)
    for ratio in (1.0, 10.0, 100.0):
        allocation = allocate_counts(available, shared_total, ratio, min_support=10)
        assert sum(allocation) == shared_total
        assert all(count <= support for count, support in zip(allocation, available))


def test_shared_total_is_maximized_before_requested_ratio_is_lowered() -> None:
    frame = pd.DataFrame(
        [
            {
                "case_id": f"{name}_{index}",
                "slide_id": f"{name}_{index}",
                "cancer_type": name,
            }
            for name, support in (("A", 200), ("B", 100), ("C", 100))
            for index in range(support)
        ]
    )

    total = cap_feasible_shared_total(
        frame,
        ["A", "B", "C"],
        min_support=20,
        is_mil=True,
        seed=1,
        independent_floor=10,
    )

    assert total == 300


def test_mil_shared_total_counts_unique_slides_not_feature_chunks() -> None:
    frame = pd.DataFrame(
        [
            {
                "case_id": f"{name}_{slide}",
                "slide_id": f"{name}_{slide}",
                "feature_path": f"{name}_{slide}_{chunk}.pt",
                "cancer_type": name,
            }
            for name in ("A", "B")
            for slide in range(30)
            for chunk in range(2)
        ]
    )

    total = cap_feasible_shared_total(
        frame,
        ["A", "B"],
        min_support=20,
        is_mil=True,
        seed=1,
        independent_floor=10,
    )

    assert total == 60


def test_bag_dataset_concatenates_all_feature_chunks_before_capping(tmp_path: Path) -> None:
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    torch.save(torch.ones(3, 4), first)
    torch.save(torch.full((4, 4), 2.0), second)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"case_id": "case", "slide_id": "slide", "cancer_type": "A", "feature_path": first},
            {"case_id": "case", "slide_id": "slide", "cancer_type": "A", "feature_path": second},
        ]
    ).to_csv(manifest, index=False)

    bag, target = BagFeatureDataset(manifest, max_instances=5)[0]

    assert target == 0
    assert len(bag) == 5
    assert bag.sum().item() == pytest.approx(32.0)


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
        return {"model": torch.nn.Linear(1, 1), "train_dataset": [0], "seed": 7, "param_config": {}}

    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation.build_training_ctx", fake_context
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._timed_fit",
        lambda _fit, _ctx: ({}, 0.0),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._run_and_record", lambda *args: None
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
    assert all(set(frame["split"]) == {"train", "validation", "test"} for frame in manifests)
    assert any(
        not manifests[0][["case_id", "split"]].equals(frame[["case_id", "split"]])
        for frame in manifests[1:]
    )


def test_freeze_uses_the_resampling_seed_family(tmp_path: Path) -> None:
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
            "split": "train" if index < 30 else "test",
        }
        for cls in ("A", "B")
        for index in range(40)
    ]
    for split_index in range(3):
        data_dir = tmp_path / "outputs" / f"split={split_index}" / "data"
        data_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(data_dir / "manifest.csv", index=False)
        (data_dir / "pilot_report.json").write_text(
            json.dumps({"definitive_floor": 10, "quotas": {"0": 1}, "excluded": False}),
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
