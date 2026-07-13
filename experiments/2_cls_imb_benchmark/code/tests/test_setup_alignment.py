from __future__ import annotations

from pathlib import Path
from argparse import Namespace

import pandas as pd
import numpy as np
import pytest
import torch
import yaml

from imbalance_benchmark.analysis.inference.permutation import paired_block_permutation_ba
from imbalance_benchmark.commands.confirm_methods import RunContext, confirm_ce
from imbalance_benchmark.commands.prepare import cmd_prepare
from imbalance_benchmark.construction import allocate_counts, max_shared_total
from imbalance_benchmark.datasets.data import BagFeatureDataset


def test_shared_total_is_feasible_for_every_requested_ratio() -> None:
    available = [100, 100, 100]
    shared_total = max_shared_total(available, min_support=10)

    assert shared_total < sum(available)
    for ratio in (1.0, 10.0, 100.0):
        allocation = allocate_counts(available, shared_total, ratio, min_support=10)
        assert sum(allocation) == shared_total
        assert all(count <= support for count, support in zip(allocation, available))


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
        "imbalance_benchmark.commands.confirm_methods.build_training_ctx", fake_context
    )
    monkeypatch.setattr(
        "imbalance_benchmark.commands.confirm_methods._timed_fit",
        lambda _fit, _ctx: ({}, 0.0),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.commands.confirm_methods._run_and_record", lambda *args: None
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


def test_two_seed_permutation_stack_is_not_mistaken_for_one_probability_matrix():
    labels = np.array([0, 1, 0, 1])
    case_ids = np.array(["P0", "P1", "P2", "P3"])
    method = np.array([[0, 1, 0, 1], [0, 1, 1, 1]])
    ce = np.array([[1, 1, 0, 1], [1, 0, 1, 1]])
    p_value = paired_block_permutation_ba(labels, method, ce, case_ids, n_classes=2)
    assert 0.0 <= p_value <= 1.0
