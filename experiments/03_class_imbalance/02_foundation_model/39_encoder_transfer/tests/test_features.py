"""Unit tests for exp-39's phase-03 extraction/audit/manifest-join mechanics.

Uses a tiny synthetic manifest and a fake embedder throughout: real UNI2-h
inference needs a GPU, gated weights, and the real datasets, all available
only on Hydra (phase 05). These tests cover the orchestration this module
owns on its own: sharding, atomic resumable extraction, cache-corruption
detection, and the cross-encoder manifest join -- not model correctness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from prevalence import DEPTH

from transfer import extract, features, manifest

_NAMES = ["clsA", "clsB"]
_G = 3
_PATIENTS_PER_CLASS = 6


def _train_rows() -> list[dict[str, Any]]:
    rows = []
    for name in _NAMES:
        for p in range(_PATIENTS_PER_CLASS):
            case = f"{name}-P{p}"
            slide = f"{case}-S0"
            for patch in range(DEPTH):
                rows.append(
                    {
                        "cancer_type": name,
                        "case_id": case,
                        "slide_id": slide,
                        "patch_id": f"{case}-p{patch:04d}",
                        "image_path": f"/img/{case}-p{patch:04d}.jpg",
                        "split": "train",
                    }
                )
    return rows


def _held_rows(kind: str) -> list[dict[str, Any]]:
    rows = []
    for i in range(2):
        case = f"held-{kind}-{i}"
        rows.append(
            {
                "cancer_type": _NAMES[i % len(_NAMES)],
                "case_id": case,
                "slide_id": f"{case}-S0",
                "patch_id": f"{case}-p0000",
                "image_path": f"/img/{case}-p0000.jpg",
                "split": kind,
            }
        )
    return rows


def _write_virchow2_cache(frame: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Fabricate a full-slide Virchow2-shaped (2560-d) tensor per slide, matching
    the real cache's convention: one multi-row tensor, feature_index picks a row."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    paths: list[str] = []
    indices: list[int] = []
    for slide_id, group in frame.groupby("slide_id", sort=False):
        path = cache_dir / f"{slide_id}.pt"
        torch.save(torch.rand(len(group), 2560), path)
        for offset, _ in enumerate(group.index):
            paths.append(str(path))
            indices.append(offset)
    frame["feature_path"] = pd.Series(dtype=object)
    frame["feature_index"] = pd.Series(dtype="Int64")
    cursor = 0
    for _, group in frame.groupby("slide_id", sort=False):
        for row_idx in group.index:
            frame.loc[row_idx, "feature_path"] = paths[cursor]
            frame.loc[row_idx, "feature_index"] = indices[cursor]
            cursor += 1
    return frame


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    full = pd.DataFrame(_train_rows() + _held_rows("validation") + _held_rows("test"))
    full = _write_virchow2_cache(full, tmp_path / "virchow2_cache")
    cfg: dict[str, Any] = {
        "paths": {"outputs": str(tmp_path / "outputs")},
        "dataset": {"name": "tcga_ut"},
    }
    return {"cfg": cfg, "full": full}


@pytest.fixture(autouse=True)
def _patch_schedule_inputs(monkeypatch, config) -> None:
    full = config["full"]
    train_only = full[full["split"] == "train"].reset_index(drop=True)

    def fake_train_identity(cfg, split_idx):
        return train_only, _NAMES

    # features.py's own copies drive requested_frame/audit_uni2h; manifest.py
    # imported its own separate references, used by join_manifest directly.
    for module in (features, manifest):
        monkeypatch.setattr(module, "_full_manifest", lambda cfg, split_idx: full)
        monkeypatch.setattr(module, "load_train_identity", fake_train_identity)
        monkeypatch.setattr(module, "patients_per_class", lambda cfg: _G)


def _calls_embed():
    calls: list[int] = []

    def embed(image_paths, dtype, device, model_cache):
        calls.append(len(image_paths))
        return torch.zeros((len(image_paths), extract.uni.FEATURE_DIM))

    return embed, calls


def test_shard_slides_disjoint_and_complete() -> None:
    ids = [f"s{i}" for i in range(7)]
    shards = [extract.shard_slides(ids, i, 3) for i in range(3)]
    flat = sorted(sum(shards, []))
    assert flat == sorted(ids)
    assert not set(shards[0]) & set(shards[1]) & set(shards[2])


def test_shard_slides_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError):
        extract.shard_slides(["a"], 2, 2)


def test_ordered_identity_sorts_by_patch_id() -> None:
    group = pd.DataFrame(
        {
            "case_id": ["c", "c"],
            "slide_id": ["s", "s"],
            "patch_id": ["p2", "p1"],
            "image_path": ["/b.jpg", "/a.jpg"],
        }
    )
    assert features.ordered_identity(group) == [
        "c\0s\0p1\0/a.jpg",
        "c\0s\0p2\0/b.jpg",
    ]


def test_requested_frame_covers_held_rows_and_subset_of_train(config) -> None:
    frame = features.requested_frame(config["cfg"])
    got = set(zip(frame["case_id"], frame["slide_id"], frame["patch_id"]))
    held = pd.DataFrame(_held_rows("validation") + _held_rows("test"))
    held_ids = set(zip(held["case_id"], held["slide_id"], held["patch_id"]))
    assert held_ids <= got
    n_train = len(config["full"][config["full"]["split"] == "train"])
    n_train_selected = len(got) - len(held_ids)
    assert 0 < n_train_selected < n_train


def test_extract_merge_audit_roundtrip(config) -> None:
    embed, calls = _calls_embed()
    extract.extract_shard(config["cfg"], 0, 1, embed_fn=embed, device=torch.device("cpu"))
    assert calls  # extraction actually ran
    extract.merge_features(config["cfg"])
    audit, index = extract.audit_uni2h(config["cfg"])
    assert audit["unresolved_missing"] == 0
    assert audit["unresolved_corrupt"] == 0
    assert audit["matched_patches"] == audit["requested_patches"]
    assert len(index) == audit["requested_patches"]


def test_restarting_a_completed_shard_skips_extraction(config) -> None:
    embed, calls = _calls_embed()
    extract.extract_shard(config["cfg"], 0, 1, embed_fn=embed, device=torch.device("cpu"))
    extract.merge_features(config["cfg"])
    calls.clear()
    extract.extract_shard(config["cfg"], 0, 1, embed_fn=embed, device=torch.device("cpu"))
    assert calls == []


def test_two_disjoint_shards_merge_to_a_complete_cache(config) -> None:
    embed, _ = _calls_embed()
    extract.extract_shard(config["cfg"], 0, 2, embed_fn=embed, device=torch.device("cpu"))
    extract.extract_shard(config["cfg"], 1, 2, embed_fn=embed, device=torch.device("cpu"))
    extract.merge_features(config["cfg"])
    audit, _ = extract.audit_uni2h(config["cfg"])
    assert audit["unresolved_missing"] == 0


def test_provenance_guard_rejects_dtype_change(config) -> None:
    embed, _ = _calls_embed()
    extract.extract_shard(config["cfg"], 0, 1, dtype="float32", embed_fn=embed, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        extract.extract_shard(
            config["cfg"], 0, 1, dtype="float16", embed_fn=embed, device=torch.device("cpu")
        )


def test_audit_flags_a_corrupted_tensor(config) -> None:
    embed, _ = _calls_embed()
    extract.extract_shard(config["cfg"], 0, 1, embed_fn=embed, device=torch.device("cpu"))
    extract.merge_features(config["cfg"])
    root = extract.uni2h_feature_root(config["cfg"])
    victim = next(root.glob("*.pt"))
    torch.save(torch.zeros(1, 3), victim)
    audit, _ = extract.audit_uni2h(config["cfg"])
    assert audit["unresolved_corrupt"] >= 1


def test_join_manifest_matches_across_encoders_after_dropping_feature_columns(config) -> None:
    embed, _ = _calls_embed()
    extract.extract_shard(config["cfg"], 0, 1, embed_fn=embed, device=torch.device("cpu"))
    extract.merge_features(config["cfg"])
    audit = manifest.run_audit(config["cfg"])
    assert audit["unresolved_missing"] == 0
    assert audit["unresolved_corrupt"] == 0
    assert audit["unresolved_mismatched"] == 0

    out_dir = manifest.split_paths(manifest.ensure_dirs(config["cfg"]), 0)["data"]
    uni_manifest = pd.read_csv(out_dir / "manifest_uni2h.csv")
    v2_manifest = pd.read_csv(out_dir / "manifest_virchow2.csv")
    drop = ["feature_path", "feature_index"]
    pd.testing.assert_frame_equal(
        uni_manifest.drop(columns=drop), v2_manifest.drop(columns=drop)
    )
    # unselected train rows carry no feature reference in either manifest.
    unselected = uni_manifest[uni_manifest["feature_path"].isna()]
    assert (unselected["split"] == "train").all()
    assert v2_manifest.loc[unselected.index, "feature_path"].isna().all()
    # every held row is resolved in both.
    held = uni_manifest[uni_manifest["split"] != "train"]
    assert held["feature_path"].notna().all()
    assert v2_manifest.loc[held.index, "feature_path"].notna().all()


def test_join_manifest_rejects_an_incomplete_identity_index(config) -> None:
    # An empty index cannot resolve any requested row's feature reference, so the
    # join must refuse rather than silently leave a required row unmatched.
    with pytest.raises(ValueError):
        manifest.join_manifest(config["cfg"], 0, {}, "uni2h")
