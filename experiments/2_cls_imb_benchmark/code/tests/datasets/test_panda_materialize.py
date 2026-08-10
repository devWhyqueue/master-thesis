from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from imbalance_benchmark.datasets import panda_materialize
from imbalance_benchmark.datasets.data.panda_grid import copy_audited_tiles
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

    audited = audit_slide(row, legacy, jpeg_mae_max=2.0)

    assert audited[["x", "y"]].values.tolist() == [
        [0, 0],
        [256, 0],
        [0, 256],
        [256, 256],
    ]
    assert audited["patch_label"].tolist() == ["cancer", "benign", "benign", "benign"]
    copied = copy_audited_tiles(audited, target)
    assert all(Path(path).is_file() for path in copied["image_path"])


def test_audit_slide_allows_known_large_official_tiffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, legacy, _ = _source(tmp_path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    audited = audit_slide(row, legacy, jpeg_mae_max=2.0)

    assert len(audited) == 4


def test_audit_slide_derives_labels_despite_legacy_label_mismatch(
    tmp_path: Path,
) -> None:
    row, legacy, _ = _source(tmp_path)
    manifest = tmp_path / "legacy.csv"
    pd.DataFrame(
        {
            "patch_id": ["13", "7", "99", "2"],
            "x": [256, 0, 256, 0],
            "y": [256, 0, 0, 256],
            "image_path": [
                str(legacy / "3.jpg"),
                str(legacy / "0.jpg"),
                str(legacy / "1.jpg"),
                str(legacy / "2.jpg"),
            ],
            "patch_label": ["cancer", "cancer", "cancer", "cancer"],
        }
    ).to_csv(manifest, index=False)

    audited = audit_slide(row, legacy, jpeg_mae_max=2.0, manifest_path=manifest)

    assert audited["patch_id"].tolist() == [
        "slide/13",
        "slide/7",
        "slide/99",
        "slide/2",
    ]
    assert audited["patch_label"].tolist() == ["benign", "cancer", "benign", "benign"]
    assert "legacy_patch_label" not in audited


@pytest.mark.parametrize("filename", ["4.jpg", "0.jpg"])
def test_audit_slide_rejects_extra_or_missing_eligible_tile(
    tmp_path: Path, filename: str
) -> None:
    row, legacy, _ = _source(tmp_path)
    if filename == "4.jpg":
        (legacy / filename).write_bytes((legacy / "0.jpg").read_bytes())
    else:
        (legacy / filename).unlink()

    with pytest.raises(ValueError, match="eligible tile coordinates"):
        audit_slide(row, legacy, jpeg_mae_max=2.0)


def test_materialize_gates_copying_on_locked_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Pool:
        def __init__(self, max_workers: int) -> None:
            del max_workers

        def __enter__(self) -> _Pool:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def map(self, function: object, jobs: object) -> object:
            return map(function, jobs)  # type: ignore[arg-type]

    config = {
        "materialize_panda": {
            "raw_root": str(tmp_path),
            "legacy_tiles_dir": str(tmp_path),
            "legacy_manifest_dir": str(tmp_path),
            "scratch_root": str(tmp_path / "scratch"),
            "shard_root": str(tmp_path / "shards"),
            "shard_mount_root": str(tmp_path / "mount"),
            "canonical_inventory_path": str(tmp_path / "inventory.csv"),
            "sidecar_path": str(tmp_path / "sidecar.json"),
        }
    }
    monkeypatch.setattr(panda_materialize, "LOCKED_SLIDES", 1)
    monkeypatch.setattr(
        panda_materialize,
        "load_slide_frame",
        lambda _: pd.DataFrame({"slide_id": ["slide"]}),
    )
    monkeypatch.setattr(
        panda_materialize,
        "audit_slide",
        lambda *_: pd.DataFrame({"slide_id": ["slide"], "patch_label": ["benign"]}),
    )
    monkeypatch.setattr(panda_materialize, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(
        panda_materialize,
        "assert_locked_counts",
        lambda *_: (_ for _ in ()).throw(ValueError("locked counts")),
    )
    monkeypatch.setattr(
        panda_materialize,
        "copy_audited_tiles",
        lambda *_: pytest.fail("copy ran before locked-count gate"),
    )

    with pytest.raises(ValueError, match="locked counts"):
        panda_materialize.materialize(config)


def test_audit_slides_uses_allocated_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[int] = []

    class _Pool:
        def __init__(self, max_workers: int) -> None:
            captured.append(max_workers)

        def __enter__(self) -> _Pool:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def map(self, function: object, jobs: object) -> object:
            return map(function, jobs)  # type: ignore[arg-type]

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setattr(panda_materialize, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(
        panda_materialize,
        "audit_slide",
        lambda row, *_: pd.DataFrame({"slide_id": [row.slide_id]}),
    )

    audited = panda_materialize._audit_slides(
        pd.DataFrame({"slide_id": ["b", "a", "c"]}),
        {
            "legacy_tiles_dir": "/tiles",
            "legacy_manifest_dir": "/manifests",
            "jpeg_mae_max": 8.0,
        },
    )

    assert captured == [2]
    assert [frame.iloc[0].slide_id for frame in audited] == ["a", "b", "c"]


def test_audit_slides_runs_a_real_process(tmp_path: Path) -> None:
    row, legacy, _ = _source(tmp_path)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    pd.DataFrame(
        {
            "patch_id": range(4),
            "x": [0, 256, 0, 256],
            "y": [0, 0, 256, 256],
            "image_path": [str(legacy / f"{index}.jpg") for index in range(4)],
        }
    ).to_csv(manifests / "slide.csv", index=False)

    audited = panda_materialize._audit_slides(
        pd.DataFrame([row]),
        {
            "legacy_tiles_dir": str(legacy.parent),
            "legacy_manifest_dir": str(manifests),
            "jpeg_mae_max": 2.0,
        },
    )

    assert len(audited) == 1
    assert audited[0].patch_label.tolist() == ["cancer", "benign", "benign", "benign"]
