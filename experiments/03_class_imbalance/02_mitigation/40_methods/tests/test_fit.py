"""Stage-one/stage-two training loop smoke tests (CPU, tiny synthetic model/data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import timm
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image

from centre.cohort import patient_rows
from prevalence.fit import arm_row_index

from mitigation.train.artifacts import Stage1Artifacts
from mitigation.train.stage1 import ArmBatch, run_stage1
from mitigation.train.stage2 import run_crt, run_disalign, run_gcl2, run_posthoc_la

_DIM = 96  # 2 * embed_dim of the tiny model below


def _fake_encoder(config: dict, device: torch.device):
    """A ``vit_tiny``-sized encoder standing in for Virchow2, same LoRA/token-output shape."""
    del config
    base = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, reg_tokens=4, num_classes=0, global_pool=""
    )
    base.train()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    lora_config = LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, target_modules=r".*blocks\.\d+\.attn\.(qkv|proj)$"
    )
    model = get_peft_model(base, lora_config).to(device)
    data_config = {
        "input_size": (3, 224, 224),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "crop_pct": 1.0,
    }
    return model, data_config, 2 * base.embed_dim


def _write_images(tmp_path: Path, n: int) -> list[str]:
    """N tiny random RGB JPEGs on disk, standing in for decoded patch images."""
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        array = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        path = tmp_path / f"patch_{i}.jpg"
        Image.fromarray(array).save(path)
        paths.append(str(path))
    return paths


def _stage1_config() -> dict:
    return {
        "mitigation": {
            "train": {"stage1": {"steps": 2, "batch_size": 4, "lr": 1e-3, "weight_decay": 0.0}}
        }
    }


def test_stage1_trains_only_lora_and_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only LoRA adapter and head parameters receive gradients; the frozen base does not."""
    monkeypatch.setattr("mitigation.train.stage1.load_lora_encoder", _fake_encoder)
    image_paths = _write_images(tmp_path, 16)
    y = np.array([i % 2 for i in range(16)], dtype=np.int64)
    counts = np.bincount(y, minlength=2).astype(np.int64)

    arm = ArmBatch(image_paths, y, counts, num_classes=2)
    output = run_stage1(_stage1_config(), torch.device("cpu"), arm, method="ce", param=0.0, seed=0)

    trainable = [p for p in output.encoder.parameters() if p.requires_grad]
    frozen = [p for p in output.encoder.parameters() if not p.requires_grad]
    assert trainable and all(p.grad is not None for p in trainable)
    assert frozen and all(p.grad is None for p in frozen)
    assert all(p.grad is not None for p in output.head.parameters())


def _fake_artifacts(n: int = 32, dim: int = 8, num_classes: int = 2, sigma: float | None = None) -> Stage1Artifacts:
    """Synthetic cached stage-one embeddings for stage-two smoke tests."""
    rng = np.random.default_rng(0)
    y = np.array([i % num_classes for i in range(n)], dtype=np.int64)
    counts = np.bincount(y, minlength=num_classes).astype(np.int64)
    head = torch.nn.Linear(dim, num_classes)
    return Stage1Artifacts(
        train_embeddings=torch.tensor(rng.normal(size=(n, dim)), dtype=torch.float16),
        train_y=y,
        val_embeddings=torch.tensor(rng.normal(size=(n, dim)), dtype=torch.float16),
        val_y=y,
        test_embeddings=torch.tensor(rng.normal(size=(n, dim)), dtype=torch.float16),
        test_y=y,
        counts=counts,
        head_state=head.state_dict(),
        is_cosine=sigma is not None,
        cosine_scale=30.0 if sigma is not None else None,
        sigma=sigma,
    )


def _stage2_config() -> dict:
    return {
        "mitigation": {
            "gcl": {"s": 30, "a": 0.9, "b": 0.09},
            "train": {"stage2": {"steps": 5, "batch_size": 8, "lr": 1e-2, "weight_decay": 0.0}},
        }
    }


def test_stage2_methods_run_on_cached_embeddings() -> None:
    """Every stage-two method fits from cached embeddings and returns (N, C) logit arrays."""
    device = torch.device("cpu")
    config = _stage2_config()
    ce_artifacts = _fake_artifacts()

    val_logits, test_logits = run_posthoc_la(ce_artifacts, 1.0, 2, device)
    assert val_logits.shape == (32, 2) and test_logits.shape == (32, 2)

    _, val_logits, test_logits = run_crt(ce_artifacts, 2, config, device, seed=0)
    assert val_logits.shape == (32, 2) and test_logits.shape == (32, 2)

    val_logits, test_logits = run_disalign(ce_artifacts, 1.0, 2, config, device, seed=0)
    assert val_logits.shape == (32, 2) and test_logits.shape == (32, 2)

    gcl_artifacts = _fake_artifacts(sigma=0.1)
    _, val_logits, test_logits = run_gcl2(gcl_artifacts, 2, config, device, seed=0)
    assert val_logits.shape == (32, 2) and test_logits.shape == (32, 2)


def test_arm_row_index_matches_patient_rows() -> None:
    """The row-index refactor selects exactly the same rows as ``centre.cohort.patient_rows``."""
    train_df = pd.DataFrame(
        [
            {"cancer_type": "cls", "case_id": "P1", "slide_id": "P1_s0", "patch_id": f"P1_p{i:03d}"}
            for i in range(80)
        ]
    )
    rows, y = arm_row_index(train_df, ["cls"], [["P1"]], [50])
    expected_rows = patient_rows(train_df, ["P1"], 50)
    assert rows.tolist() == expected_rows
    assert len(y) == len(rows) == 50
    assert set(y.tolist()) == {0}


def test_pool_tokens_passes_through_already_pooled_output() -> None:
    """UNI2-h returns (B, D) pooled features; Virchow2 (B, T, D) tokens become CLS + mean."""
    from mitigation.encoder import pool_tokens

    pooled = torch.ones(3, 8)
    assert pool_tokens(pooled) is pooled
    assert pool_tokens(torch.ones(3, 9, 8)).shape == (3, 16)


def test_inline_stage2_fits_from_cache_then_drops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``inline_stage2`` a ce fit runs stage two on the cache, deletes it, records stage one last."""
    from types import SimpleNamespace

    from mitigation import fit

    events: list[str] = []
    shard = SimpleNamespace(classes=["a", "b"])
    monkeypatch.setattr(fit, "load_shard", lambda config, split_idx: shard)
    monkeypatch.setattr(fit, "run_dir", lambda *args: tmp_path)
    monkeypatch.setattr(fit, "resolve_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        fit, "train_arm", lambda *args: (["x"], np.zeros(1, dtype=np.int64), np.array([1, 0]))
    )
    monkeypatch.setattr(fit, "run_stage1", lambda *args: None)
    monkeypatch.setattr(fit, "_eval_embeddings", lambda *args: None)

    def save(out_dir: Path, *args) -> None:
        (out_dir / "stage1.pt").write_text("cache")
        events.append("save")

    def stage2(*args) -> None:
        assert (tmp_path / "stage1.pt").exists()
        events.append("stage2")

    monkeypatch.setattr(fit, "_save_stage1_artifacts", save)
    monkeypatch.setattr(fit, "run_stage2_from_source", stage2)
    monkeypatch.setattr(fit, "_record_stage1_run", lambda *args: events.append("record"))
    monkeypatch.setattr(fit, "patients_per_class", lambda config: 20)

    fit.run_fit_stage1({"mitigation": {"inline_stage2": True}}, 0, 0, "r1", "ce", None)

    assert events == ["save", "stage2", "record"]
    assert not (tmp_path / "stage1.pt").exists()
