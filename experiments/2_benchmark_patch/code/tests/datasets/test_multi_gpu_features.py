from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import imbalance_benchmark.datasets.features.attach as feature_attach
from imbalance_benchmark.datasets import features
from imbalance_benchmark.datasets.features.cache_manifest import (
    merge_pending_slides,
    record_pending_slide,
    save_tensor_atomic,
)


def test_balanced_assignments_are_duplicate_free_and_keep_slide_order() -> None:
    work = [
        ("large", ["a"] * 6, ["a"] * 6),
        ("small", ["b"], ["b"]),
        ("medium", ["c"] * 4, ["c"] * 4),
    ]

    assignments = feature_attach._balanced_assignments(work, 2)

    assert sorted(slide_id for worker in assignments for slide_id, *_ in worker) == [
        "large",
        "medium",
        "small",
    ]
    assert [sum(len(paths) for _, paths, _ in worker) for worker in assignments] == [
        6,
        5,
    ]
    assert [identities for worker in assignments for _, _, identities in worker] == [
        ["a"] * 6,
        ["c"] * 4,
        ["b"],
    ]


def test_multi_gpu_request_falls_back_to_serial_without_visible_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_attach.torch.cuda, "is_available", lambda: False)

    assert feature_attach._gpu_worker_count(4) == 1


def test_multi_gpu_request_uses_each_visible_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_attach.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(feature_attach.torch.cuda, "device_count", lambda: 3)

    assert feature_attach._gpu_worker_count(4) == 3


def test_loader_workers_divide_slurm_cpus_between_gpu_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")

    assert feature_attach._loader_workers(4) == 7


def test_pending_slide_is_recovered_without_reextracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        features,
        "extract_slide_features",
        lambda paths, *_args: calls.append(paths) or torch.ones(len(paths), 2560),
    )
    frame = pd.DataFrame({"slide_id": ["s1", "s1"], "image_path": ["a.jpg", "b.jpg"]})
    root = tmp_path / "features"
    options = feature_attach._extraction_options({}, None)
    work = feature_attach._slide_work(frame)
    feature_attach._prepare_feature_cache(root, options)
    feature_attach._extract_one(work[0], root, options, {})

    enriched = feature_attach.attach_extracted_features(frame, root)

    assert calls == [["a.jpg", "b.jpg"]]
    assert enriched["feature_index"].tolist() == [0, 1]
    assert not list((root / ".feature_cache_pending").glob("*.json"))


def test_pending_records_merge_in_deterministic_slide_id_order(tmp_path: Path) -> None:
    root = tmp_path / "features"
    expected = {}
    for slide_id in ("s2", "s1"):
        path = root / f"{slide_id}.pt"
        identities = [f"{slide_id}\0{slide_id}.jpg"]
        path.parent.mkdir(parents=True, exist_ok=True)
        save_tensor_atomic(torch.ones(1, 2560), path)
        record_pending_slide(root, slide_id, path, identities, 1)
        expected[slide_id] = (path, identities)

    merge_pending_slides(root, expected)

    assert list(json.loads((root / "feature_cache_manifest.json").read_text())) == [
        "s1",
        "s2",
    ]


def test_parallel_worker_failure_reaches_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedProcess:
        exitcode = 1

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class FailedContext:
        def Process(self, **_kwargs: object) -> FailedProcess:
            return FailedProcess()

    monkeypatch.setattr(
        feature_attach.multiprocessing, "get_context", lambda *_args: FailedContext()
    )

    with pytest.raises(RuntimeError, match="workers failed"):
        feature_attach._extract_parallel(
            [("s1", ["a.jpg"], ["a\0a.jpg"])], tmp_path, {}, 2
        )
