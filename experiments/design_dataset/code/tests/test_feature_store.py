import torch

from data.feature_store import (
    DEFAULT_FEATURE_DIR,
    load_feature_row,
    load_slide_features,
    SlideFeatureStore,
    verify_feature_store,
)


def test_load_patch_feature_selects_sorted_patch(tmp_path) -> None:
    slide_id = "TCGA-XX-0001"
    chunk = torch.stack([torch.randn(2560), torch.randn(2560), torch.randn(2560)])
    torch.save(chunk, tmp_path / f"{slide_id}_0.pt")
    store = SlideFeatureStore(str(tmp_path))
    vector = store.load_patch_feature(slide_id, ["1_2", "0_9", "1_0"], "1_0")
    assert vector.shape == (2560,)
    assert torch.allclose(vector, load_feature_row(str(tmp_path / f"{slide_id}_0.pt"), 1))


def test_patch_sort_key_ordering(tmp_path) -> None:
    slide_id = "TCGA-XX-0001"
    torch.save(torch.tensor([0.0, 1.0, 2.0]), tmp_path / f"{slide_id}_0.pt")
    store = SlideFeatureStore(str(tmp_path))
    assert store.patch_index(["1_2", "0_9", "0_0"], "0_9") == 1


def test_verify_feature_store_on_synthetic_dir(tmp_path) -> None:
    slide_id = "TCGA-XX-0001"
    torch.save(torch.randn(3, 2560), tmp_path / f"{slide_id}_0.pt")
    report = verify_feature_store(str(tmp_path), expected_dim=2560)
    assert report["dim_matches"]
    assert report["feature_dir"] == str(tmp_path)
    assert report["layout"] == "chunked_slide_tensors"
    assert DEFAULT_FEATURE_DIR.endswith("raw")


def test_load_slide_features_normalizes_vector(tmp_path) -> None:
    path = tmp_path / "TCGA-XX-0001_0.pt"
    torch.save(torch.randn(2560), path)
    tensor = load_slide_features(str(path))
    assert tensor.shape == (1, 2560)
