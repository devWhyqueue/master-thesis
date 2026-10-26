"""Frozen-encoder CUDA: encoder immutability, dynamic augmentation, candidate reset, micro-batching, cache gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import timm
import torch
from imbalance_benchmark.common import RUN_RECORD_NAME
from peft import LoraConfig, get_peft_model

from mitigation import TrainHParams
from mitigation.data import ImageShard
from mitigation.images import normalize_batch
from mitigation.train import ArmBatch
from mitigation.train import stage1_step

from lora_ablation import frozen_cuda
from lora_ablation.heads import extraction_transform
from lora_ablation.frozen_cuda import (
    CudaTask,
    check_cached_features,
    encoder_fingerprint,
    fit_cuda_draw,
    run_frozen_cuda,
)
from lora_ablation.preflight import audit_inputs, write_inventory
from lora_ablation.resume import AUDIT_SIDECAR

from world import CLASSES, DIM, G, build_world, make_manifest, patch_freeze_loader

DEVICE = torch.device("cpu")
MEAN = STD = (0.5, 0.5, 0.5)
N, SIZE = 16, 224


def _encoder() -> torch.nn.Module:
    """A vit_tiny stand-in for Virchow2 with the same register-token layout (pooled width 2 * 192)."""
    torch.manual_seed(0)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, reg_tokens=4, num_classes=0, global_pool="")
    return model


def _config(steps: int = 4, batch: int = 8) -> dict[str, Any]:
    return {
        "dataset": {"name": "bracs"},
        "protocol": {"init_seed_base": 10000},
        "mitigation": {
            "cuda": {"gamma": 0.6, "check_size": 2, "max_strength": 10},
            "train": {"stage1": {"steps": steps, "batch_size": batch, "lr": 1e-2, "weight_decay": 0.0}},
        },
    }


def _arm(value: int | None = None) -> tuple[ArmBatch, torch.Tensor]:
    rng = np.random.default_rng(0)
    images = torch.from_numpy(rng.integers(0, 255, (N, 3, SIZE, SIZE), dtype=np.uint8))
    if value is not None:
        images = torch.full_like(images, value)
    y = np.array([i % 2 for i in range(N)], dtype=np.int64)
    return ArmBatch([], y, np.bincount(y).astype(np.int64), 2), images


def _train(encoder: torch.nn.Module, images: torch.Tensor, arm: ArmBatch, param: float, config: dict | None = None) -> torch.nn.Module:
    task = CudaTask(encoder, {"mean": MEAN, "std": STD}, arm, images)
    return run_frozen_cuda(config or _config(), DEVICE, task, param, (0, 1))


def test_encoder_stays_frozen_while_head_learns(monkeypatch) -> None:
    for name in ("load_lora_encoder",):
        monkeypatch.setattr(f"mitigation.encoder.{name}", lambda *a, **k: pytest.fail("LoRA loader called"))
    monkeypatch.setattr("peft.get_peft_model", lambda *a, **k: pytest.fail("adapter built"))
    encoder = _encoder()
    arm, images = _arm()
    before = encoder_fingerprint(encoder)
    initial = frozen_cuda._new_head(2 * encoder.embed_dim, 2, 1).weight.detach().clone()
    head = _train(encoder, images, arm, 0.5)
    assert encoder_fingerprint(encoder) == before  # bitwise unchanged
    assert not encoder.training
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert not torch.equal(head.weight.detach(), initial) and head.weight.grad is not None
    assert not any("lora" in name for name, _ in encoder.named_parameters())


def test_adapter_encoder_is_rejected() -> None:
    adapted = get_peft_model(_encoder(), LoraConfig(r=2, lora_alpha=4, target_modules=r".*blocks\.\d+\.attn\.(qkv|proj)$"))
    arm, images = _arm()
    with pytest.raises(ValueError, match="adapter"):
        _train(adapted, images, arm, 0.5)


def test_curriculum_uses_augmented_images(monkeypatch) -> None:
    encoder = _encoder()
    arm, images = _arm(value=100)
    seen: list[torch.Tensor] = []
    real_forward = encoder.forward
    monkeypatch.setattr(encoder, "forward", lambda x: (seen.append(x.detach().clone()), real_forward(x))[1])
    checks: list[tuple] = []
    monkeypatch.setattr(stage1_step, "update_levels", lambda levels, rows, predict, *rest: checks.append(rest[:3]) or levels)
    monkeypatch.setattr(stage1_step, "cuda_batch", lambda raw, levels, p, max_s, g: torch.full_like(raw, 155))
    _train(encoder, images, arm, 1.0)
    augmented = normalize_batch(torch.full((1, 3, SIZE, SIZE), 155, dtype=torch.uint8), MEAN, STD)[0, 0, 0, 0]
    original = normalize_batch(torch.full((1, 3, SIZE, SIZE), 100, dtype=torch.uint8), MEAN, STD)[0, 0, 0, 0]
    assert seen and all(torch.allclose(x, augmented.expand_as(x)) and not torch.allclose(x, original.expand_as(x)) for x in seen)
    assert checks == [(0.6, 2, 10)] * 2  # source schedule: one check per epoch of 2 optimizer steps


def test_candidates_reset_head_and_curriculum(tmp_path, monkeypatch) -> None:
    patch_freeze_loader(monkeypatch)
    config, shard, calls = _draw_world(tmp_path, monkeypatch)
    states: list[int] = []
    real_init = stage1_step._CudaState.__init__

    def counting(self, model, device, params):
        real_init(self, model, device, params)
        assert int(self.levels.sum()) == 0  # fresh curriculum
        states.append(1)

    monkeypatch.setattr(stage1_step._CudaState, "__init__", counting)
    fit_cuda_draw(config, 0, 0)
    assert len(states) == 3
    assert calls["encoder"] == 1 and calls["decode"] == 1  # one model and one decode per draw
    assert len({tuple(w.flatten().tolist()) for w in calls["initial"]}) == 1  # identical fresh initialization
    assert len(calls["final"]) == 3  # one trained head per probability
    for p in ("cuda0.25", "cuda0.5", "cuda1"):
        run = shard.paths["results"] / f"r100_{p}" / "draw_0"
        meta = json.loads((run / AUDIT_SIDECAR).read_text())
        assert meta["micro_batch"] == 64 and meta["device"] == "cpu" and meta["checkpoint"]["revision"]
        assert (run / RUN_RECORD_NAME).exists()
    calls["encoder"] = calls["decode"] = 0
    fit_cuda_draw(config, 0, 0)  # complete outputs are reused without loading anything
    assert calls["encoder"] == calls["decode"] == 0


def test_microbatch_gradient_equivalence(monkeypatch) -> None:
    arm, images = _arm()
    heads = []
    for micro in (4, 1000):
        monkeypatch.setattr(frozen_cuda, "CUDA_MICRO_BATCH", micro)
        heads.append(_train(_encoder(), images, arm, 0.0).weight.detach())  # p_aug = 0: deterministic
    torch.testing.assert_close(heads[0], heads[1], atol=1e-5, rtol=1e-4)


def test_cache_consistency_gate() -> None:
    direct = torch.randn(32, 8)
    check_cached_features(direct.half(), direct)  # fp16 round-off passes
    with pytest.raises(ValueError, match="shape"):
        check_cached_features(direct[:31], direct)
    for bad in (direct + 0.5 * torch.randn(32, 8), -direct):  # cosine / relative L2 violations
        with pytest.raises(ValueError):
            check_cached_features(bad, direct)
    zero = direct.clone()
    zero[3] = 0
    with pytest.raises(ValueError, match="zero"):
        check_cached_features(zero, direct)
    nonfinite = direct.clone()
    nonfinite[5, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        check_cached_features(nonfinite, direct)
    near = direct * 1.012  # relative L2 error 0.012 > 0.01 although cosine is 1
    with pytest.raises(ValueError, match="L2"):
        check_cached_features(near, direct)


def _draw_world(tmp_path: Path, monkeypatch) -> tuple[dict, ImageShard, dict]:
    """A full config/inventory/shard whose images are tiny tensors and whose encoder is a vit_tiny."""
    config = build_world(tmp_path)
    write_inventory(config, audit_inputs(config))
    manifest = make_manifest(tmp_path / "features", tmp_path / "images")
    parts = {n: manifest.query("split == @n").reset_index(drop=True) for n in ("train", "validation", "test")}
    ident = {n: parts[n][["case_id", "slide_id", "cancer_type"]] for n in ("validation", "test")}
    shard = ImageShard(parts["train"], parts["validation"], parts["test"], ident["validation"], ident["test"], CLASSES,
                       {"results": tmp_path / "new_out" / "split=0" / "results"})
    calls: dict[str, Any] = {"encoder": 0, "decode": 0, "initial": [], "final": []}
    encoder = _encoder()
    real_head, real_run = frozen_cuda._new_head, frozen_cuda.run_frozen_cuda

    def features(df: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(1)
        return rng.normal(size=(len(df), 2 * encoder.embed_dim)).astype(np.float16)

    def load(device):
        calls["encoder"] += 1
        return encoder, {"mean": MEAN, "std": STD}

    def decode(paths, transform):
        calls["decode"] += len(paths) > 32  # the 32-row cache check is separate from the training decode
        return torch.from_numpy(np.random.default_rng(2).integers(0, 255, (len(paths), 3, SIZE, SIZE), dtype=np.uint8))

    def new_head(dim, k, seed):
        head = real_head(dim, k, seed)
        calls["initial"].append(head.weight.detach().clone())
        return head

    monkeypatch.setattr(frozen_cuda, "train_hparams", lambda cfg, stage: TrainHParams(8, 4, 1e-2, 0.0))
    monkeypatch.setattr(frozen_cuda, "_load_frozen_encoder", load)
    monkeypatch.setattr(frozen_cuda, "decode_images", decode)
    monkeypatch.setattr(frozen_cuda, "extraction_transform", lambda cfg: None)
    monkeypatch.setattr(frozen_cuda, "load_shard", lambda cfg, split: shard)
    monkeypatch.setattr(frozen_cuda, "load_features_for_df", features)
    monkeypatch.setattr(frozen_cuda, "check_cached_features", lambda cached, direct: None)
    monkeypatch.setattr(frozen_cuda, "_new_head", new_head)
    def run(*args, **kwargs):
        head = real_run(*args, **kwargs)
        calls["final"].append(head.weight.detach().clone())
        return head

    monkeypatch.setattr(frozen_cuda, "run_frozen_cuda", run)
    return config, shard, calls


def test_decode_matches_extraction_recipe() -> None:
    """Bicubic resize (timm's recipe), not exp-40's bilinear: the cache was extracted that way."""
    from PIL import Image
    from timm.data.transforms_factory import create_transform

    config = {"input_size": (3, 224, 224), "interpolation": "bicubic", "mean": MEAN, "std": STD, "crop_pct": 1.0}
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
    ours = normalize_batch(extraction_transform(config)(image)[None], MEAN, STD)[0]
    reference = create_transform(**config)(image)
    assert torch.allclose(ours, reference, atol=1e-5)
