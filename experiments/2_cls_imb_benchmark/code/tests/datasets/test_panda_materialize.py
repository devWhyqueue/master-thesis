from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from imbalance_benchmark.datasets import panda_materialize
from imbalance_benchmark.datasets.data.panda import grid as panda_grid
from imbalance_benchmark.datasets.data.panda import partials as panda_partials
from imbalance_benchmark.datasets.data.panda.slide_audit import copy_audited_tiles
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

    audited = audit_slide(row, legacy, jpeg_mae_max=2.0, jpeg_workers=2)

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


def test_band_starts_covers_full_height_in_512_row_bands() -> None:
    # 768 rows -> one full 512-row band, then a clamped 256-row trailing band;
    # this is the same boundary math the fused audit loop reads by (one
    # source read per band, not per 256-row tile stripe).
    assert list(panda_grid._band_starts(768, 512)) == [(0, 512), (512, 256)]
    assert list(panda_grid._band_starts(512, 512)) == [(0, 512)]
    assert list(panda_grid._band_starts(513, 512)) == [(0, 512)]


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


def _materialize_config(tmp_path: Path) -> dict[str, object]:
    return {
        "materialize_panda": {
            "raw_root": str(tmp_path),
            "legacy_tiles_dir": str(tmp_path),
            "legacy_manifest_dir": str(tmp_path),
            "scratch_root": str(tmp_path / "scratch"),
            "shard_root": str(tmp_path / "shards"),
            "shard_mount_root": str(tmp_path / "mount"),
            "canonical_inventory_path": str(
                tmp_path / "materialize" / "canonical_inventory.csv"
            ),
            "sidecar_path": str(tmp_path / "materialize" / "sidecar.json"),
        }
    }


def test_combine_gates_writing_inventory_on_locked_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _materialize_config(tmp_path)
    monkeypatch.setattr(panda_materialize, "LOCKED_SLIDES", 1)
    monkeypatch.setattr(
        panda_materialize,
        "load_slide_frame",
        lambda _: pd.DataFrame({"slide_id": ["slide"]}),
    )
    monkeypatch.setattr(
        panda_materialize,
        "assert_locked_counts",
        lambda *_: (_ for _ in ()).throw(ValueError("locked counts")),
    )
    cfg = panda_materialize.materialize_config(config)
    frame = pd.DataFrame({"slide_id": ["slide"], "patch_label": ["benign"]})
    panda_materialize.write_audit_partial(cfg, 0, frame, {"slide": {"image": "x"}})

    with pytest.raises(ValueError, match="locked counts"):
        panda_materialize.combine(config, shard_count=1)

    inventory_path, raw_path = panda_partials.combined_paths(cfg)
    assert not inventory_path.exists()
    assert not raw_path.exists()


def test_audit_partial_round_trips_and_rejects_tampering(tmp_path: Path) -> None:
    cfg = panda_materialize.materialize_config(_materialize_config(tmp_path))
    frame = pd.DataFrame(
        {
            "slide_id": ["slide"],
            "case_id": ["slide"],
            "slide_label": ["ISUP0"],
            "provider": ["karolinska"],
            "has_mask": [True],
            "patch_id": ["slide/0"],
            "patch_label": ["benign"],
            "legacy_image_path": ["/legacy/0.jpg"],
            "sha256": ["abc"],
            "x": [0],
            "y": [0],
            "level": [0],
            "tile_size": [256],
            "tissue_fraction_min": [0.35],
            "tissue_intensity_threshold": [210.0],
        }
    )
    raw_hashes = {"slide": {"image": "abc", "mask": None}}

    assert not panda_materialize.audit_partial_done(cfg, 0)
    panda_materialize.write_audit_partial(cfg, 0, frame, raw_hashes)
    assert panda_materialize.audit_partial_done(cfg, 0)

    read_frame, read_raw = panda_materialize.read_audit_partial(cfg, 0)
    pd.testing.assert_frame_equal(read_frame, frame)
    assert read_raw == raw_hashes

    tiles_path, _ = panda_partials._audit_partial_paths(cfg, 0)
    tiles_path.write_text(tiles_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        panda_materialize.audit_partial_done(cfg, 0)
    with pytest.raises(RuntimeError):
        panda_materialize.read_audit_partial(cfg, 0)


def test_audit_shard_slices_cover_every_slide_exactly_once() -> None:
    slides = pd.Series([f"slide-{i:04d}" for i in range(101)])
    count = 8
    covered: list[str] = []
    for index in range(count):
        covered.extend(slides.iloc[index::count].tolist())

    assert sorted(covered) == sorted(slides.tolist())
    assert len(covered) == len(slides)


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
