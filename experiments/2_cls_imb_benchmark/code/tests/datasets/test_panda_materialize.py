from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from imbalance_benchmark.datasets.panda_materialize import audit_slide


def _source(tmp_path: Path) -> tuple[pd.Series, Path, Path]:
    raw = tmp_path / "raw"
    images, masks = raw / "train_images", raw / "train_label_masks"
    images.mkdir(parents=True)
    masks.mkdir()
    slide_id = "slide"
    source = np.full((512, 512, 3), 100, dtype=np.uint8)
    Image.fromarray(source).save(images / f"{slide_id}.tiff")
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:256, :256] = 2
    Image.fromarray(mask).save(masks / f"{slide_id}_mask.tiff")
    tiles = tmp_path / "legacy" / slide_id
    tiles.mkdir(parents=True)
    for index, (x, y) in enumerate(((0, 0), (256, 0), (0, 256), (256, 256))):
        Image.fromarray(source[y : y + 256, x : x + 256]).save(tiles / f"{index}.jpg")
    return (
        pd.Series(
            {
                "slide_id": slide_id,
                "provider": "karolinska",
                "slide_label": "ISUP0",
                "image_path": str(images / f"{slide_id}.tiff"),
                "mask_path": str(masks / f"{slide_id}_mask.tiff"),
                "has_mask": True,
            }
        ),
        tiles,
        tmp_path / "target",
    )


def test_audit_slide_recomputes_complete_grid_and_source_fidelity(
    tmp_path: Path,
) -> None:
    row, legacy, target = _source(tmp_path)

    audited = audit_slide(row, legacy, target, jpeg_mae_max=2.0)

    assert audited[["x", "y"]].values.tolist() == [
        [0, 0],
        [256, 0],
        [0, 256],
        [256, 256],
    ]
    assert audited["patch_label"].tolist() == ["cancer", "benign", "benign", "benign"]
    assert all(Path(path).is_file() for path in audited["image_path"])


def test_audit_slide_allows_known_large_official_tiffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, legacy, target = _source(tmp_path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    audited = audit_slide(row, legacy, target, jpeg_mae_max=2.0)

    assert len(audited) == 4


@pytest.mark.parametrize("filename", ["4.jpg", "0.jpg"])
def test_audit_slide_rejects_extra_or_missing_eligible_tile(
    tmp_path: Path, filename: str
) -> None:
    row, legacy, target = _source(tmp_path)
    if filename == "4.jpg":
        (legacy / filename).write_bytes((legacy / "0.jpg").read_bytes())
    else:
        (legacy / filename).unlink()

    with pytest.raises(ValueError, match="eligible tile coordinates"):
        audit_slide(row, legacy, target, jpeg_mae_max=2.0)
