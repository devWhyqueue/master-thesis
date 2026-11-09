"""Frozen-encoder SGD: CE through exp-45's CUDA path without augmentation, and the per-draw fitter."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
import timm
import torch
from imbalance_benchmark.common import RUN_RECORD_NAME

from analyze.paths import run_dir
from mitigation import TrainHParams
from mitigation.data import ImageShard
from mitigation.train import ArmBatch
from mitigation.train import stage1_step

from lora_ablation import frozen_cuda
from lora_ablation.frozen_cuda import CudaTask, encoder_fingerprint, run_frozen_cuda

from intended import sgd_paths
from intended import frozen_sgd
from intended.frozen_sgd import SIDECAR_NAME, fit_sgd_draw

from intended_world import CLASSES, make_config, make_manifest

DEVICE = torch.device("cpu")
MEAN = STD = (0.5, 0.5, 0.5)
N, SIZE = 16, 224


def _encoder() -> torch.nn.Module:
    torch.manual_seed(0)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, reg_tokens=4, num_classes=0, global_pool="")


def _config() -> dict[str, Any]:
    return {
        "mitigation": {
            "cuda": {"gamma": 0.6, "check_size": 2, "max_strength": 10},
            "train": {"stage1": {"steps": 4, "batch_size": 8, "lr": 1e-2, "weight_decay": 0.0}},
        },
    }


def _task(encoder: torch.nn.Module) -> CudaTask:
    images = torch.from_numpy(np.random.default_rng(0).integers(0, 255, (N, 3, SIZE, SIZE), dtype=np.uint8))
    y = np.array([i % 2 for i in range(N)], dtype=np.int64)
    arm = ArmBatch([], y, np.bincount(y).astype(np.int64), 2)
    return CudaTask(encoder, {"mean": MEAN, "std": STD}, arm, images)


def test_ce_method_trains_head_only() -> None:
    encoder = _encoder()
    before = encoder_fingerprint(encoder)
    initial = frozen_cuda._new_head(2 * encoder.embed_dim, 2, 1).weight.detach().clone()
    head = run_frozen_cuda(_config(), DEVICE, _task(encoder), 0.0, (0, 1), method="ce")
    assert encoder_fingerprint(encoder) == before
    assert not torch.equal(head.weight.detach(), initial)


def test_ce_method_no_augmentation(monkeypatch) -> None:
    monkeypatch.setattr(stage1_step, "cuda_batch", lambda *a, **k: pytest.fail("augmented a CE batch"))
    monkeypatch.setattr(stage1_step, "update_levels", lambda *a, **k: pytest.fail("curriculum check in CE"))
    run_frozen_cuda(_config(), DEVICE, _task(_encoder()), 1.0, (0, 1), method="ce")


def test_default_is_cuda(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(stage1_step, "cuda_batch", lambda raw, *a: calls.append(1) or raw)
    run_frozen_cuda(_config(), DEVICE, _task(_encoder()), 1.0, (0, 1))
    assert calls


def _sgd_world(tmp_path, monkeypatch) -> tuple[dict, dict]:
    config = make_config(tmp_path, "bracs")
    manifest = make_manifest(tmp_path / "features", tmp_path / "images")
    parts = {n: manifest.query("split == @n").reset_index(drop=True) for n in ("train", "validation", "test")}
    ident = {n: parts[n][["case_id", "slide_id", "cancer_type"]] for n in ("validation", "test")}
    shard = ImageShard(parts["train"], parts["validation"], parts["test"], ident["validation"], ident["test"], CLASSES, {})
    encoder = _encoder()
    calls: dict[str, Any] = {"encoder": 0, "decode": 0, "methods": []}

    def load(device):
        calls["encoder"] += 1
        return encoder, {"mean": MEAN, "std": STD}

    def decode(paths, transform):
        calls["decode"] += 1
        return torch.from_numpy(np.random.default_rng(2).integers(0, 255, (len(paths), 3, SIZE, SIZE), dtype=np.uint8))

    def features(df: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(1).normal(size=(len(df), 2 * encoder.embed_dim)).astype(np.float16)

    real_run = frozen_cuda.run_frozen_cuda

    def run(*args, **kwargs):
        calls["methods"].append(kwargs.get("method", "cuda"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(frozen_cuda, "train_hparams", lambda cfg, stage: TrainHParams(8, 4, 1e-2, 0.0))
    monkeypatch.setattr(frozen_cuda, "_load_frozen_encoder", load)
    monkeypatch.setattr(frozen_cuda, "_check_cache", lambda *a: None)
    monkeypatch.setattr(frozen_cuda, "run_frozen_cuda", run)
    monkeypatch.setattr(frozen_sgd, "decode_images", decode)
    monkeypatch.setattr(frozen_sgd, "extraction_transform", lambda cfg: None)
    monkeypatch.setattr(frozen_sgd, "load_shard", lambda cfg, split: shard)
    monkeypatch.setattr(frozen_sgd, "load_features_for_df", features)
    return config, calls


def test_fit_sgd_draw_writes_three_kinds(tmp_path, monkeypatch) -> None:
    config, calls = _sgd_world(tmp_path, monkeypatch)
    fit_sgd_draw(config, 0, 3)
    paths = sgd_paths(config)[0]
    expected = [("r1", "ce", None), ("r100", "ce", None)] + [("r1", "cuda", p) for p in (0.25, 0.5, 1.0)]
    for arm, method, param in expected:
        directory = run_dir(paths, arm, method, param, 3)
        assert (directory / RUN_RECORD_NAME).is_file()
        sidecar = json.loads((directory / SIDECAR_NAME).read_text())
        assert sidecar["seeds"] == [3, 10003] and sidecar["method"] == method
    assert sorted(calls["methods"]) == ["ce", "ce", "cuda", "cuda", "cuda"]
    assert calls["encoder"] == 1 and calls["decode"] == 2  # one encoder load, one decode per arm
    calls["encoder"] = calls["decode"] = 0
    fit_sgd_draw(config, 0, 3)  # complete outputs are reused without loading anything
    assert calls["encoder"] == calls["decode"] == 0


def test_changed_encoder_raises(tmp_path, monkeypatch) -> None:
    config, _ = _sgd_world(tmp_path, monkeypatch)
    real_run = frozen_cuda.run_frozen_cuda

    def tamper(config, device, task, *args, **kwargs):
        head = real_run(config, device, task, *args, **kwargs)
        next(task.encoder.parameters()).data.add_(1.0)
        return head

    monkeypatch.setattr(frozen_cuda, "run_frozen_cuda", tamper)
    with pytest.raises(ValueError, match="encoder changed"):
        fit_sgd_draw(config, 0, 0)
