from __future__ import annotations

import pandas as pd
import torch

from imbalance_benchmark.datasets import features as feature_lib
from imbalance_benchmark.datasets.features import (
    SlideFeatureStore,
    load_feature_row,
    load_slide_features,
    patch_sort_key,
    _virchow2_pool,
)


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


def test_slide_feature_store_resolves_sorted_patch(tmp_path) -> None:
    slide_id = "TCGA-XX-0001"
    chunk = torch.stack([torch.randn(2560), torch.randn(2560), torch.randn(2560)])
    torch.save(chunk, tmp_path / f"{slide_id}_0.pt")
    store = SlideFeatureStore(str(tmp_path))

    vector = store.load_patch_feature(slide_id, ["1_2", "0_9", "1_0"], "1_0")

    assert vector.shape == (2560,)
    assert torch.allclose(vector, load_feature_row(str(tmp_path / f"{slide_id}_0.pt"), 1))


def test_slide_feature_store_spans_multiple_chunks(tmp_path) -> None:
    slide_id = "TCGA-XX-0002"
    torch.save(torch.stack([torch.full((4,), float(i)) for i in range(30)]), tmp_path / f"{slide_id}_0.pt")
    torch.save(torch.stack([torch.full((4,), float(30 + i)) for i in range(5)]), tmp_path / f"{slide_id}_1.pt")
    patch_ids = [f"0_{i}" for i in range(35)]
    store = SlideFeatureStore(str(tmp_path))

    vector = store.load_patch_feature(slide_id, patch_ids, "0_32")

    assert torch.allclose(vector, torch.full((4,), 32.0))


def test_attach_extracted_features_writes_one_tensor_per_slide(tmp_path, monkeypatch) -> None:
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
