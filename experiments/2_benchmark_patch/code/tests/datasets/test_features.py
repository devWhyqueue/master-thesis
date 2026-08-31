from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from imbalance_benchmark.commands import prepare
from imbalance_benchmark.datasets import features
from imbalance_benchmark.datasets import features as feature_lib
from imbalance_benchmark.datasets.features import (
    _virchow2_pool,
    load_feature_model,
    load_feature_row,
    load_slide_features,
    patch_sort_key,
)
from imbalance_benchmark.datasets.features import cache as feature_cache
from imbalance_benchmark.datasets.features.cache import (
    bank_index,
    bank_is_cpu,
    feature_rows,
    reset_feature_bank,
)
from imbalance_benchmark.datasets.data.patch import ImbalanceDataset


def test_upstream_wsi_tiles_require_auditable_realization_fields() -> None:
    from imbalance_benchmark.datasets.bracs.audit import validate_tile_manifest
    from imbalance_benchmark.datasets.features.panda_audit import (
        validate_tile_inventory,
    )

    with pytest.raises(ValueError, match="audit"):
        validate_tile_manifest(
            pd.DataFrame({"slide_id": ["s"], "image_path": ["tile.jpg"]}),
            expected_slides=1,
        )
    with pytest.raises(ValueError, match="audit"):
        validate_tile_inventory(
            pd.DataFrame({"slide_id": ["s"]}),
            {"s": pd.DataFrame({"image_path": ["tile.jpg"]})},
            pd.DataFrame({"slide_id": ["s"]}),
            expected_slides=1,
        )


def test_frozen_feature_reuse_verifies_revision_order_rows_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        features,
        "extract_slide_features",
        lambda paths, *_args, **_kwargs: torch.ones(len(paths), 2560),
    )
    frame = pd.DataFrame(
        {
            "slide_id": ["s1", "s1"],
            "patch_id": ["p0", "p1"],
            "image_path": ["p0.jpg", "p1.jpg"],
        }
    )
    root = tmp_path / "features"
    features.attach_extracted_features(frame, root)
    provenance = json.loads((root / "feature_provenance.json").read_text())
    assert provenance["encoder_revision"]
    assert provenance["weights_sha256"]

    with pytest.raises(ValueError, match="patch order"):
        features.attach_extracted_features(frame.iloc[::-1], root)

    tensor_path = root / "s1.pt"
    torch.save(torch.ones(1, 2560), tensor_path)
    with pytest.raises(ValueError, match="row count|hash"):
        features.attach_extracted_features(frame, root)


def test_load_feature_model_loads_verified_safetensors_explicitly(
    tmp_path, monkeypatch
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.touch()
    observed = {}

    class Model:
        pretrained_cfg = {}

        def load_state_dict(self, state_dict):
            observed["state_dict"] = state_dict

        def eval(self):
            return self

        def to(self, device):
            observed["device"] = device
            return self

    def create_model(name, **kwargs):
        observed["name"] = name
        observed["pretrained"] = kwargs["pretrained"]
        return Model()

    monkeypatch.setattr(feature_lib, "resolve_feature_snapshot", lambda *_: tmp_path)
    monkeypatch.setattr(feature_lib.timm, "create_model", create_model)
    monkeypatch.setattr(
        feature_lib, "load_safetensors", lambda path: {"weights": str(path)}
    )
    monkeypatch.setattr(
        feature_lib, "resolve_data_config", lambda *_args, **_kwargs: {}
    )
    transform = object()
    monkeypatch.setattr(feature_lib, "create_transform", lambda **_: transform)

    model, returned_transform = load_feature_model("model", torch.device("cpu"))

    assert observed == {
        "name": f"local-dir:{tmp_path.as_posix()}",
        "pretrained": False,
        "state_dict": {"weights": str(weights)},
        "device": torch.device("cpu"),
    }
    assert isinstance(model, Model)
    assert returned_transform is transform


def test_patch_sort_key_orders_by_region_then_index() -> None:
    items = ["1_2", "0_9", "1_0", "0_0"]
    assert sorted(items, key=patch_sort_key) == ["0_0", "0_9", "1_0", "1_2"]


def test_virchow2_pool_concatenates_cls_and_mean_patch_tokens() -> None:
    # (batch=1, tokens=7, dim=4): token 0 is CLS, tokens 1-4 are register tokens
    # dropped by the [:, 5:] slice, tokens 5-6 are the two patch tokens averaged.
    output = torch.zeros(1, 7, 4)
    output[0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    output[0, 5] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    output[0, 6] = torch.tensor([3.0, 3.0, 3.0, 3.0])

    pooled = _virchow2_pool(output)

    assert pooled.shape == (1, 8)
    assert torch.allclose(pooled[0, :4], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(pooled[0, 4:], torch.tensor([2.0, 2.0, 2.0, 2.0]))


def test_load_slide_features_normalizes_single_vector(tmp_path) -> None:
    path = tmp_path / "slide_0.pt"
    torch.save(torch.randn(2560), path)
    tensor = load_slide_features(str(path))
    assert tensor.shape == (1, 2560)


def test_load_feature_row_requires_index_for_multirow(tmp_path) -> None:
    path = tmp_path / "slide_0.pt"
    torch.save(torch.randn(3, 2560), path)
    try:
        load_feature_row(str(path))
    except ValueError as error:
        assert "provide feature_index" in str(error)
    else:
        raise AssertionError("Expected multi-row tensor to require an index.")
    vector = load_feature_row(str(path), 1)
    assert vector.shape == (2560,)


def test_attach_extracted_features_writes_one_tensor_per_slide(
    tmp_path, monkeypatch
) -> None:
    calls: list[list[str]] = []

    def fake_extract(image_paths, *args, **kwargs):
        calls.append(list(image_paths))
        return torch.randn(len(image_paths), 2560)

    monkeypatch.setattr(feature_lib, "extract_slide_features", fake_extract)
    frame = pd.DataFrame(
        {
            "slide_id": ["s1", "s1", "s2"],
            "image_path": ["s1_a.jpg", "s1_b.jpg", "s2_a.jpg"],
        }
    )

    enriched = feature_lib.attach_extracted_features(frame, tmp_path / "features")

    assert len(calls) == 2  # one extraction call per slide
    assert enriched.loc[0, "feature_path"] == enriched.loc[1, "feature_path"]
    assert enriched.loc[0, "feature_index"] == 0
    assert enriched.loc[1, "feature_index"] == 1
    assert enriched.loc[2, "feature_index"] == 0
    assert (tmp_path / "features" / "s1.pt").exists()
    assert (tmp_path / "features" / "s2.pt").exists()

    # Re-running with existing tensors on disk must not re-extract.
    feature_lib.attach_extracted_features(frame, tmp_path / "features")
    assert len(calls) == 2


def test_feature_extraction_rejects_a_non_virchow2_encoder() -> None:
    with pytest.raises(ValueError, match="Virchow2"):
        features.resolve_feature_provenance({"model_name": "resnet50"})


def test_prepare_validates_encoder_for_precomputed_feature_manifests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        prepare,
        "build_manifest",
        lambda _: pd.DataFrame({"feature_path": ["cached.pt"]}),
    )

    with pytest.raises(ValueError, match="Virchow2"):
        prepare._base_manifest(
            {
                "dataset": {"name": "precomputed"},
                "feature_extraction": {"model_name": "resnet50"},
            },
            {"data": tmp_path},
        )


def test_bank_stays_on_cpu_without_cuda(tmp_path: Path) -> None:
    path = tmp_path / "slide_0.pt"
    torch.save(torch.randn(1, 8), path)

    rows = feature_rows([str(path)], [None])

    assert bank_is_cpu()
    assert bank_index(rows).device.type == "cpu"


def test_target_bank_device_env_override_bypasses_auto_sizing() -> None:
    assert feature_cache._target_bank_device({"IMB_FEATURE_BANK_DEVICE": "cpu"}, 1) == (
        torch.device("cpu")
    )
    assert feature_cache._target_bank_device(
        {"IMB_FEATURE_BANK_DEVICE": "cuda"}, 1
    ) == (torch.device("cuda"))


def test_target_bank_device_auto_falls_back_to_cpu_without_cuda() -> None:
    assert feature_cache._target_bank_device({}, 1) == torch.device("cpu")


def test_target_bank_device_auto_respects_the_free_memory_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_cache.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(feature_cache.torch.cuda, "mem_get_info", lambda: (1000, 1000))

    assert feature_cache._target_bank_device({}, 850) == torch.device("cuda")
    assert feature_cache._target_bank_device({}, 851) == torch.device("cpu")


def test_feature_bank_reserves_a_split_manifest_without_concatenation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"slide_{index}.pt"
        torch.save(torch.full((1, 2), index, dtype=torch.float16), path)
        paths.append(str(path))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "slide_id": [f"slide_{index}" for index in range(4)],
            "cancer_type": ["A", "B", "A", "B"],
            "feature_path": paths,
            "split": ["validation", "validation", "train", "train"],
        }
    ).to_csv(manifest, index=False)
    monkeypatch.setattr(
        feature_cache.torch,
        "cat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no cat")),
    )

    validation = ImbalanceDataset(manifest, "validation", class_names=["A", "B"])
    training = ImbalanceDataset(manifest, "train", class_names=["A", "B"])

    assert validation.rows.tolist() == [0, 1]
    assert training.rows.tolist() == [2, 3]
    assert feature_cache._BANK is not None
    assert feature_cache._BANK.shape == (4, 2)
    assert feature_cache._BANK.dtype == torch.float16
    assert feature_cache._BANK.device.type == "cpu"
    assert torch.equal(
        bank_index(torch.tensor([3, 0])),
        torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
    )
    assert bank_index(training.rows).dtype == torch.float32

    reset_feature_bank()
    assert bank_is_cpu()
    with pytest.raises(RuntimeError, match="empty"):
        bank_index(validation.rows)


def test_feature_bank_keeps_its_reservation_for_a_later_smaller_manifest(
    tmp_path: Path,
) -> None:
    """A per-condition frame is far smaller than the split frame banked before it."""
    reset_feature_bank()
    paths = []
    for index in range(4):
        path = tmp_path / f"slide_{index}.pt"
        torch.save(torch.full((1, 2), index, dtype=torch.float16), path)
        paths.append(str(path))
    columns = {
        "slide_id": [f"slide_{i}" for i in range(4)],
        "cancer_type": list("ABAB"),
    }
    split = tmp_path / "manifest.csv"
    pd.DataFrame({**columns, "feature_path": paths}).to_csv(split, index=False)
    condition = tmp_path / "manifest_balanced.csv"
    pd.DataFrame(
        {
            "slide_id": ["slide_0"],
            "cancer_type": ["A"],
            "feature_path": [paths[0]],
        }
    ).to_csv(condition, index=False)

    ImbalanceDataset(split, None, class_names=["A", "B"])
    banked = ImbalanceDataset(condition, None, class_names=["A", "B"])

    assert banked.rows.tolist() == [0]
    assert feature_cache._BANK is not None
    assert feature_cache._BANK.shape == (4, 2)
    reset_feature_bank()


def test_feature_cache_rejects_metadata_from_a_different_encoder_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame({"slide_id": ["s1"], "image_path": ["s1.jpg"]})
    monkeypatch.setattr(
        features,
        "extract_slide_features",
        lambda *_args, **_kwargs: torch.ones(1, 2560),
    )
    root = tmp_path / "features"
    features.attach_extracted_features(frame, root, {"dtype": "float16"})

    with pytest.raises(ValueError, match="provenance"):
        features.attach_extracted_features(frame, root, {"dtype": "float32"})
