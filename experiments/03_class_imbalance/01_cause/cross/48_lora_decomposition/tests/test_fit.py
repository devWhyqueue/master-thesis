"""Seeded four-arm fits on a tiny LoRA ViT: pairing, reproducibility, provenance, and resume."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import timm
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image

from mitigation.train.stage1 import ArmBatch, Stage1Extras, run_stage1

from lora_decomposition import fit, protocol
from lora_decomposition.protocol import ARMS, load_protocol_config

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _tiny_encoder(config: dict, device: torch.device):
    del config
    base = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, img_size=32, reg_tokens=4,
        num_classes=0, global_pool="",
    )
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    lora = LoraConfig(r=2, lora_alpha=4, target_modules=r".*blocks\.\d+\.attn\.(qkv|proj)$")
    data_config = {"input_size": (3, 32, 32), "mean": (0.5,) * 3, "std": (0.5,) * 3, "crop_pct": 1.0}
    return get_peft_model(base, lora).to(device), data_config, 2 * base.embed_dim


def _images(root: Path, prefix: str, n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        path = root / f"{prefix}_{i}.png"
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


def _eval_frame(paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = ["a" if i % 2 == 0 else "b" for i in range(len(paths))]
    cases = [f"P{i % 4}{label}" for i, label in enumerate(labels)]
    df = pd.DataFrame({"image_path": paths, "cancer_type": labels, "case_id": cases})
    identity = pd.DataFrame({"case_id": cases, "slide_id": [f"{c}_s" for c in cases], "cancer_type": labels})
    return df, identity


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A smoke-budget BRACS config rooted in ``tmp_path`` with a synthetic two-class shard."""
    base = load_protocol_config(_CONFIGS / "bracs.yaml")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setattr(protocol, "EXPERIMENT_DIR", tmp_path)
    monkeypatch.setitem(protocol.TRAIN_ROWS, "bracs", 16)
    monkeypatch.setattr(fit, "_IDENTITY_CACHE", {})
    monkeypatch.setattr("mitigation.train.stage1.load_lora_encoder", _tiny_encoder)
    train = _images(tmp_path / "img", "train", 16, 0)
    val_df, val_identity = _eval_frame(_images(tmp_path / "img", "val", 8, 1))
    test_df, test_identity = _eval_frame(_images(tmp_path / "img", "test", 8, 2))
    shard = SimpleNamespace(
        classes=["a", "b"], train_df=pd.DataFrame({"image_path": train}),
        val_df=val_df, test_df=test_df, val_identity=val_identity, test_identity=test_identity,
    )
    monkeypatch.setattr(fit, "load_shard", lambda config, split_idx: shard)

    def train_arm(shard_, split_idx, draw_idx, g, data_arm):
        n_a = 8 if data_arm == "r1" else 14
        y = np.array([0] * n_a + [1] * (16 - n_a), dtype=np.int64)
        return train, y, np.bincount(y, minlength=2)

    monkeypatch.setattr(fit, "train_arm", train_arm)

    def make_config(root: str, steps: int = 2) -> dict:
        config = copy.deepcopy(base)
        config["paths"]["outputs"] = str(tmp_path / "outputs" / "smoke" / root)
        config["smoke"] = {"steps": steps}
        config["mitigation"]["train"]["stage1"].update(steps=steps, batch_size=8)
        return config

    monkeypatch.setitem(protocol._RECIPES, "bracs", {"batch_size": 8, "steps": 2, "lr": 0.0001, "weight_decay": 0.0001})
    return SimpleNamespace(make_config=make_config)


def _provenance(out_dir: Path) -> dict:
    return json.loads((out_dir / fit.PROVENANCE_NAME).read_text(encoding="utf-8"))


def _logits(out_dir: Path) -> np.ndarray:
    return np.load(out_dir / "eval_arrays.npz")["test_logits"]


def test_four_arms_share_initialization_and_paired_sampling(world) -> None:
    """Same start in every arm; B/F and C/R share index streams; order and block seeds behave."""
    forward = world.make_config("forward")
    dirs = {arm: fit.run_fit(forward, 0, 0, arm) for arm in ARMS}
    prov = {arm: _provenance(d) for arm, d in dirs.items()}
    audits = {arm: p["training_audit"] for arm, p in prov.items()}
    assert len({a["initialization_sha256"] for a in audits.values()}) == 1
    for left, right in (("B", "F"), ("C", "R")):
        assert audits[left]["sample_sequence_sha256"] == audits[right]["sample_sequence_sha256"]
        assert audits[left]["unique_rows_per_class"] == audits[right]["unique_rows_per_class"]
    identity = {arm: p["training_identity_sha256"] for arm, p in prov.items()}
    assert identity["B"] == identity["F"] != identity["C"] == identity["R"]
    np.testing.assert_allclose(prov["F"]["class_weights"], [14 / 16 * 2, 2 / 16 * 2])
    assert prov["F"]["target_counts"] == prov["R"]["data_counts"] == [14, 2]
    assert all(fit.validate_run(forward, 0, 0, arm) for arm in ARMS)

    reverse = world.make_config("reverse")
    reverse_dirs = {arm: fit.run_fit(reverse, 0, 0, arm) for arm in reversed(ARMS)}
    for arm in ARMS:
        assert _provenance(reverse_dirs[arm])["training_audit"] == audits[arm]
        np.testing.assert_allclose(_logits(reverse_dirs[arm]), _logits(dirs[arm]), atol=1e-5)

    other_block = _provenance(fit.run_fit(forward, 0, 1, "B"))
    assert other_block["seed"] != prov["B"]["seed"]
    assert other_block["training_audit"]["initialization_sha256"] != audits["B"]["initialization_sha256"]


def test_unit_weights_train_exactly_like_plain_ce(world, tmp_path: Path) -> None:
    """All-one fixed weights (arms B and R) reproduce unweighted CE under identical seeds."""
    paths = _images(tmp_path / "unit", "u", 8, 5)
    arm = ArmBatch(paths, np.array([0, 1] * 4), np.array([4, 4]), num_classes=2)
    config = world.make_config("unit")
    heads = []
    for weights in (None, np.ones(2)):
        fit.seed_everything(7)
        out = run_stage1(config, torch.device("cpu"), arm, "ce", 0.0, 7, extras=Stage1Extras(class_weights=weights))
        heads.append(out.head.weight.detach().clone())
    torch.testing.assert_close(heads[0], heads[1], rtol=0, atol=0)


def test_resume_rejects_incomplete_and_incompatible_runs(world, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing sidecar/arrays refit; a changed recipe or reordered predictions raise."""
    config = world.make_config("resume")
    out_dir = fit.run_fit(config, 0, 0, "B")
    monkeypatch.setattr(fit, "run_stage1", lambda *a, **k: pytest.fail("valid run refit"))
    assert fit.run_fit(config, 0, 0, "B") == out_dir

    sidecar = out_dir / fit.PROVENANCE_NAME
    saved = sidecar.read_bytes()
    sidecar.unlink()
    assert fit.validate_run(config, 0, 0, "B") is False
    sidecar.write_bytes(saved)
    arrays = out_dir / "eval_arrays.npz"
    saved_arrays = arrays.read_bytes()
    arrays.unlink()
    assert fit.validate_run(config, 0, 0, "B") is False
    arrays.write_bytes(saved_arrays)
    assert fit.validate_run(config, 0, 0, "B") is True

    changed = world.make_config("resume", steps=3)
    with pytest.raises(ValueError, match="config_sha256"):
        fit.validate_run(changed, 0, 0, "B")

    data = dict(np.load(arrays))
    data["test_labels"] = data["test_labels"][::-1].copy()
    data["test_preds"] = data["test_preds"][::-1].copy()
    data["test_logits"] = data["test_logits"][::-1].copy()
    np.savez_compressed(arrays, **data)
    with pytest.raises(ValueError, match=str(out_dir.name)):
        fit.validate_run(config, 0, 0, "B")
