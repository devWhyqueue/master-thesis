"""Cached-feature fitting: row identity, no encoder access, source recipe, CE reuse, RNG, and resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from imbalance_benchmark.common import RUN_RECORD_NAME, read_run_record

from mitigation.data import ImageShard, eval_arm, train_arm

from lora_ablation import fit, heads, init_seed
from lora_ablation.fit import AUDIT_SIDECAR, HEAD_NAME, fit_cpu_split, fit_candidate, frozen_artifacts
from lora_ablation.preflight import audit_inputs, write_inventory

from world import CLASSES, DIM, G, build_world, make_manifest, patch_freeze_loader

LA_TAUS = (0.5, 1.0)
POSTHOC_TAUS = (0.5, 1.0, 1.5, 2.0)


def _features(df: pd.DataFrame) -> np.ndarray:
    """Deterministic fake features whose first column is the row's unique feature index."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(int(df["feature_index"].max()) + 1, DIM)).astype(np.float16)
    base[:, 0] = np.arange(len(base), dtype=np.float16)
    return base[df["feature_index"].to_numpy()]


def _shard(tmp_path: Path) -> ImageShard:
    manifest = make_manifest(tmp_path / "features", tmp_path / "images")
    parts = {n: manifest.query("split == @n").reset_index(drop=True) for n in ("train", "validation", "test")}
    identity = {n: parts[n][["case_id", "slide_id", "cancer_type"]] for n in ("validation", "test")}
    results = tmp_path / "new_out" / "split=0" / "results"
    return ImageShard(
        parts["train"], parts["validation"], parts["test"],
        identity["validation"], identity["test"], CLASSES, {"results": results},
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    patch_freeze_loader(monkeypatch)
    config = build_world(tmp_path)
    write_inventory(config, audit_inputs(config))
    shard = _shard(tmp_path)
    monkeypatch.setattr(fit, "load_shard", lambda cfg, split: shard)
    monkeypatch.setattr(fit, "load_features_for_df", _features)
    monkeypatch.setattr(heads, "load_features_for_df", _features)
    return config, shard


def _eval_features(shard: ImageShard) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(_features(shard.val_df)), torch.from_numpy(_features(shard.test_df))


def _run(shard: ImageShard, label: str, draw: int = 0) -> Path:
    return shard.paths["results"] / f"r100_{label}" / f"draw_{draw}"


def test_rows_match_image_pipeline(setup) -> None:
    _, shard = setup
    artifacts = frozen_artifacts(shard, 0, 3, G, _eval_features(shard))
    paths, y, counts = train_arm(shard, 0, 3, G, "r100")
    by_path = shard.train_df.set_index("image_path")["feature_index"]
    assert [int(by_path[p]) for p in paths] == artifacts.train_embeddings[:, 0].float().int().tolist()
    assert artifacts.train_y.tolist() == y.tolist()
    assert artifacts.counts.tolist() == counts.tolist()
    assert artifacts.val_y.tolist() == eval_arm(shard.val_df, CLASSES)[1].tolist()
    assert artifacts.train_embeddings.dtype == torch.float16
    assert artifacts.sigma is None and not artifacts.is_cosine


def test_cached_methods_never_load_encoder(setup, monkeypatch) -> None:
    config, shard = setup

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("encoder / images / first-stage / adapted cache touched")

    for module, name in (
        ("mitigation.encoder", "load_lora_encoder"),
        ("mitigation.encoder", "_load_base"),
        ("mitigation.train.stage1", "run_stage1"),
        ("mitigation.train.artifacts", "load_stage1"),
        ("mitigation.images", "decode_images"),
        ("imbalance_benchmark.datasets.features", "load_feature_model"),
    ):
        monkeypatch.setattr(f"{module}.{name}", boom)
    fit_cpu_split(config, 0, 0)
    for label in ("ce", *(f"la{t:g}" for t in LA_TAUS), *(f"posthoc_la{t:g}" for t in POSTHOC_TAUS)):
        assert (_run(shard, label) / RUN_RECORD_NAME).exists(), label
    assert not (_run(shard, "cuda0.5")).exists()


def test_stage1_recipe_and_loss(setup, monkeypatch) -> None:
    config, shard = setup
    seen: dict[str, Any] = {}
    real_loop, real_la, real_weights = heads.run_training_steps, heads.la_loss, heads.sample_weights

    def loop(weights, hparams, optimizer, scheduler, generator, step_fn):
        seen.setdefault("loops", []).append((hparams, type(optimizer), optimizer.defaults, scheduler.T_max, type(scheduler)))
        return real_loop(weights, hparams, optimizer, scheduler, generator, step_fn)

    def la(logits, y, log_prior, tau):
        seen.setdefault("la", []).append((log_prior.clone(), tau))
        return real_la(logits, y, log_prior, tau)

    def weights(y, k, kind, param=0.0, **kwargs):
        seen.setdefault("weights", []).append((kind, param))
        return real_weights(y, k, kind, param, **kwargs)

    monkeypatch.setattr(heads, "run_training_steps", loop)
    monkeypatch.setattr(heads, "la_loss", la)
    monkeypatch.setattr(heads, "sample_weights", weights)
    artifacts = frozen_artifacts(shard, 0, 0, G, _eval_features(shard))
    head, _, _ = fit_candidate(config, artifacts, "la", 1.0, 0, 0)
    assert type(head) is torch.nn.Linear and head.in_features == DIM
    ((hparams, opt_type, defaults, t_max, sched_type),) = seen["loops"]
    assert (hparams.batch_size, hparams.steps, hparams.lr, hparams.weight_decay) == (256, 600, 1e-4, 1e-4)
    assert hparams.micro_batch is None and opt_type is torch.optim.AdamW
    assert (defaults["lr"], defaults["weight_decay"]) == (1e-4, 1e-4)
    assert sched_type is torch.optim.lr_scheduler.CosineAnnealingLR and t_max == 600
    assert seen["weights"] == [("power", 0.0)]  # empirical row sampling
    expected = torch.log(torch.tensor(artifacts.counts / artifacts.counts.sum(), dtype=torch.float32))
    assert all(torch.equal(p, expected) and tau == 1.0 for p, tau in seen["la"]) and len(seen["la"]) == 600
    with pytest.raises(ValueError):
        fit_candidate(config, artifacts, "cuda", 0.5, 0, 0)


def test_posthoc_reuses_one_ce_checkpoint(setup, monkeypatch) -> None:
    config, shard = setup
    methods: list[str] = []
    real = fit.fit_candidate
    monkeypatch.setattr(fit, "fit_candidate", lambda c, a, m, p, s, d: (methods.append(m), real(c, a, m, p, s, d))[1])
    scored: list[dict] = []
    real_post = fit.run_posthoc_la
    monkeypatch.setattr(fit, "run_posthoc_la", lambda a, t, k, dev: (scored.append(a.head_state), real_post(a, t, k, dev))[1])
    fit_cpu_split(config, 0, 0)
    assert methods.count("ce") == 1 and methods.count("la") == 2  # one CE fit for four strengths
    saved = torch.load(_run(shard, "ce") / HEAD_NAME, weights_only=True)
    assert len(scored) == 4 and all(torch.equal(s["weight"], saved["weight"]) for s in scored)
    digest = hashlib.sha256((_run(shard, "ce") / HEAD_NAME).read_bytes()).hexdigest()
    sidecar = json.loads((_run(shard, "posthoc_la1.5") / AUDIT_SIDECAR).read_text())
    assert sidecar["ce_head_sha256"] == digest

    methods.clear(), scored.clear()
    fit_cpu_split(config, 0, 0)
    assert methods == [] and scored == []  # complete, matching outputs are reused

    (_run(shard, "ce") / HEAD_NAME).unlink()  # missing weights invalidate CE
    fit_cpu_split(config, 0, 0)
    assert methods == ["ce"] and scored == []  # deterministic refit gives identical weights, so scores stay valid

    methods.clear()
    torch.save({"weight": torch.zeros(3, 3), "bias": torch.zeros(3)}, _run(shard, "ce") / HEAD_NAME)  # incompatible
    fit_cpu_split(config, 0, 0)
    assert methods == ["ce"] and scored == []

    sidecar_path = _run(shard, "posthoc_la1") / AUDIT_SIDECAR  # a post-hoc record bound to other CE weights is stale
    sidecar_path.write_text(json.dumps({**json.loads(sidecar_path.read_text()), "ce_head_sha256": "other"}))
    methods.clear()
    fit_cpu_split(config, 0, 0)
    assert methods == [] and len(scored) == 1


def test_rng_and_feature_reuse(setup, monkeypatch) -> None:
    config, shard = setup
    artifacts = frozen_artifacts(shard, 0, 0, G, _eval_features(shard))
    torch.manual_seed(7)
    before = torch.get_rng_state()
    first = fit_candidate(config, artifacts, "ce", None, 0, 0)[1]
    assert torch.equal(torch.get_rng_state(), before)  # isolated RNG context
    np.testing.assert_array_equal(first, fit_candidate(config, artifacts, "ce", None, 0, 0)[1])
    assert not np.array_equal(first, fit_candidate(config, artifacts, "ce", None, 2, 5)[1])
    assert init_seed(2, 5) == 10205

    loads: list[bool] = []

    def counting(df: pd.DataFrame) -> np.ndarray:
        loads.append(bool((df["split"] != "train").all()))
        return _features(df)

    monkeypatch.setattr(fit, "load_features_for_df", counting)
    monkeypatch.setattr(heads, "load_features_for_df", counting)
    monkeypatch.setattr(
        fit, "fit_candidate",
        lambda c, a, m, p, s, d: (torch.nn.Linear(DIM, 2), np.zeros((len(a.val_y), 2)), np.zeros((len(a.test_y), 2))),
    )
    fit_cpu_split(config, 0, None)
    assert sum(loads) == 2 and len(loads) == 2 + 10  # eval once per split; training once per draw
    assert read_run_record(_run(shard, "posthoc_la2", 9), splits=("test",)) is not None


def test_resume_rejects_changed_fingerprint(setup) -> None:
    config, shard = setup
    fit_cpu_split(config, 0, 0)
    (_run(shard, "la0.5") / AUDIT_SIDECAR).unlink()  # incomplete directory is refit, never skipped
    fit_cpu_split(config, 0, 0)
    assert (_run(shard, "la0.5") / AUDIT_SIDECAR).exists()
    sidecar = _run(shard, "la1") / AUDIT_SIDECAR
    sidecar.write_text(json.dumps({**json.loads(sidecar.read_text()), "audit_fingerprint": "stale"}))
    with pytest.raises(ValueError, match="fingerprint"):
        fit_cpu_split(config, 0, 0)


def test_changed_configuration_blocks_fit(setup) -> None:
    config, _ = setup
    config["prevalence"]["patients_per_class"] = G + 1
    with pytest.raises(ValueError, match="inventory"):
        fit_cpu_split(config, 0, 0)


def test_missing_inventory_blocks_fit(setup) -> None:
    config, _ = setup
    (Path(config["paths"]["outputs"]) / "inventory.json").unlink()
    with pytest.raises(ValueError, match="inventory"):
        fit_cpu_split(config, 0, 0)
