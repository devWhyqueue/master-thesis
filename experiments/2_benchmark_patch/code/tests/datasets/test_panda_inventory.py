from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from imbalance_benchmark.datasets.feature_provenance import FEATURE_DIM
from imbalance_benchmark.datasets.features.cache_manifest import (
    record_pending_slide,
    save_tensor_atomic,
)
from imbalance_benchmark.datasets.features.panda_inventory import (
    reduce_feature_inventory,
)

_CONFIG = {
    "feature_extraction": {"dtype": "float16"},
    "feature_inventory_path": "",
}


def _write_slide(root: Path, slide_id: str, tensor: torch.Tensor) -> list[str]:
    path = root / f"{slide_id}.pt"
    identities = [f"p{slide_id}\0{slide_id}.jpg"]
    save_tensor_atomic(tensor, path)
    record_pending_slide(root, slide_id, path, identities, len(tensor))
    return identities


def test_reduce_accepts_float32_storage_under_float16_compute_dtype(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    root.mkdir()
    _write_slide(root, "s1", torch.ones(1, FEATURE_DIM, dtype=torch.float32))
    frame = pd.DataFrame(
        {"slide_id": ["s1"], "patch_id": ["ps1"], "image_path": ["s1.jpg"]}
    )
    config = {**_CONFIG, "feature_inventory_path": str(tmp_path / "inventory.json")}

    reduce_feature_inventory(config, frame, root)

    assert Path(config["feature_inventory_path"]).is_file()


def test_reduce_rejects_non_float32_storage(tmp_path: Path) -> None:
    root = tmp_path / "features"
    root.mkdir()
    _write_slide(root, "s1", torch.ones(1, FEATURE_DIM, dtype=torch.float16))
    frame = pd.DataFrame(
        {"slide_id": ["s1"], "patch_id": ["ps1"], "image_path": ["s1.jpg"]}
    )
    config = {**_CONFIG, "feature_inventory_path": str(tmp_path / "inventory.json")}

    with pytest.raises(ValueError, match="dtype differs"):
        reduce_feature_inventory(config, frame, root)
