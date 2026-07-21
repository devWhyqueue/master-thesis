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
from imbalance_benchmark.datasets.data import BagFeatureDataset
from imbalance_benchmark.manifest.construction_helpers import (
    _retains_fixed_pool,
    cap_feasible_shared_total,
)
from imbalance_benchmark.manifest.pilot import run_pilot_seed
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.modeling.workflows.confirmation import RunContext, confirm_ce

def test_pilot_construction_and_initialization_seeds_are_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[int] = []

    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot.build_patch_pilot_manifest",
        lambda *_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot.evaluate_pilot_candidate",
        lambda *args, **kwargs: (
            observed.append(kwargs["initialization_seed"]) or (0.5, [0.5])
        ),
    )

    construction_seed = 17
    run_pilot_seed(
        pd.DataFrame(),
        ["A"],
        [5],
        construction_seed,
        object(),
        torch.device("cpu"),
        1,
        False,
        tmp_path,
        1,
        initialization_seed=101,
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

def test_bag_instance_cap_uses_the_frozen_selection_seed(tmp_path: Path) -> None:
    feature = tmp_path / "slide.pt"
    torch.save(torch.arange(24, dtype=torch.float32).reshape(12, 2), feature)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case",
                "slide_id": "slide",
                "cancer_type": "A",
                "feature_path": feature,
            }
        ]
    ).to_csv(manifest, index=False)

    first, _ = BagFeatureDataset(manifest, max_instances=4, instance_selection_seed=1)[
        0
    ]
    repeat, _ = BagFeatureDataset(manifest, max_instances=4, instance_selection_seed=1)[
        0
    ]
    second, _ = BagFeatureDataset(manifest, max_instances=4, instance_selection_seed=2)[
        0
    ]

    assert torch.equal(first, repeat)
    assert not torch.equal(first, second)

def test_shared_total_keeps_all_naturally_balanced_support() -> None:
    available = [100, 100, 100]
    shared_total = max_shared_total(available, min_support=10)

    assert shared_total == sum(available)
    for ratio in (1.0, 10.0, 100.0):
        allocation = allocate_counts(available, shared_total, ratio, min_support=10)
        assert sum(allocation) == shared_total
        assert all(count <= support for count, support in zip(allocation, available))

def test_shared_total_uses_one_extra_example_when_balanced_counts_can_differ_by_one() -> (
    None
):
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

    assert total == 301

def test_retains_fixed_pool_accepts_a_selection_within_the_larger_pool() -> None:
    """The pool is sized to the largest count any condition needs for a
    class, so most conditions select a strict subset of it -- requiring the
    selection to contain the *entire* pool instead can only ever hold when a
    condition's count happens to equal that maximum, i.e. almost never."""
    pool = pd.DataFrame({"case_id": ["p1", "p2", "p3"], "slide_id": ["s1", "s2", "s3"]})
    smaller_condition = pool.iloc[:2]

    assert _retains_fixed_pool(smaller_condition, pool)
    assert not _retains_fixed_pool(
        pd.DataFrame({"case_id": ["p9"], "slide_id": ["s9"]}), pool
    )

def test_cap_feasible_shared_total_uses_a_bounded_number_of_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A linear scan over a wide range with an expensive per-step check can
    take hours; binary search must stay logarithmic in the range width."""
    from imbalance_benchmark.manifest import construction_helpers as ch

    threshold, calls = 12345, []

    def fake_cap_feasible(ctx, assignments, total):
        calls.append(total)
        return total <= threshold

    monkeypatch.setattr(ch, "class_support_counts", lambda df, is_mil: {"A": 1, "B": 1})
    monkeypatch.setattr(ch, "max_shared_total", lambda available, min_support: 100_000)
    monkeypatch.setattr(ch, "_cap_feasible", fake_cap_feasible)

    total = ch.cap_feasible_shared_total(
        pd.DataFrame(), ["A", "B"], min_support=10, is_mil=False, seed=1
    )

    assert total == threshold
    assert len(calls) < 40  # log2(100000) ~ 17; a linear scan would need ~87600+

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
