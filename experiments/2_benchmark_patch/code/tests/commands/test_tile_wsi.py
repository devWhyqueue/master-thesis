from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest
import yaml

from imbalance_benchmark.commands import prepare
from imbalance_benchmark.common import compute_sha256


def _config_path(tmp_path: Path) -> Path:
    tile_root = tmp_path / "wsi_tiles"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"outputs": str(tmp_path / "outputs")},
                "dataset": {
                    "name": "bracs",
                    "regime": "wsi",
                    "root": str(tmp_path / "raw"),
                    "wsi_tile_root": str(tile_root),
                    "wsi_tile_manifest": str(tile_root / "tile_manifest.csv"),
                    "expected_wsi_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _fake_frame(slide_id: str, tile_dir: Path) -> pd.DataFrame:
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_path = tile_dir / f"{slide_id}.png"
    tile_path.write_bytes(slide_id.encode())
    return pd.DataFrame(
        {
            "slide_id": [slide_id],
            "image_path": [str(tile_path)],
            "magnification": ["20x"],
            "tile_size": [256],
            "x": [0],
            "y": [0],
            "otsu_foreground_fraction": [0.5],
            "grayscale_std": [10.0],
            "canny_edge_count": [1],
            "tissue_neighbors": [2],
            "sha256": [compute_sha256(tile_path)],
        }
    )


def test_cmd_tile_wsi_tiles_only_its_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _config_path(tmp_path)
    monkeypatch.setattr(
        prepare,
        "discover_slides",
        lambda _root: {"BRACS_1": Path("a.svs"), "BRACS_2": Path("b.svs")},
    )
    calls: list[str] = []
    tile_dir = tmp_path / "tiles"
    monkeypatch.setattr(
        prepare,
        "tile_slide",
        lambda _path, slide_id, _tile_root: calls.append(slide_id)
        or _fake_frame(slide_id, tile_dir),
    )

    prepare.cmd_tile_wsi(
        argparse.Namespace(config=str(config_path), slide_index=1, shard_size=1)
    )

    assert calls == ["BRACS_2"]
    partial = tmp_path / "wsi_tiles" / "_partials" / "BRACS_2.csv"
    assert partial.is_file()


def test_cmd_tile_wsi_rejects_out_of_range_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _config_path(tmp_path)
    monkeypatch.setattr(prepare, "discover_slides", lambda _root: {"BRACS_1": Path("a.svs")})

    with pytest.raises(ValueError, match="slide-index"):
        prepare.cmd_tile_wsi(
            argparse.Namespace(config=str(config_path), slide_index=5, shard_size=1)
        )


def test_cmd_tile_wsi_reduce_concatenates_and_validates_partials(tmp_path: Path) -> None:
    config_path = _config_path(tmp_path)
    partial_dir = tmp_path / "wsi_tiles" / "_partials"
    partial_dir.mkdir(parents=True)
    tile_dir = tmp_path / "tiles"
    for slide_id in ("BRACS_1", "BRACS_2"):
        _fake_frame(slide_id, tile_dir).to_csv(partial_dir / f"{slide_id}.csv", index=False)

    prepare.cmd_tile_wsi_reduce(argparse.Namespace(config=str(config_path)))

    manifest = pd.read_csv(tmp_path / "wsi_tiles" / "tile_manifest.csv")
    assert sorted(manifest["slide_id"]) == ["BRACS_1", "BRACS_2"]


def test_cmd_tile_wsi_reduce_raises_when_no_partials_exist(tmp_path: Path) -> None:
    config_path = _config_path(tmp_path)

    with pytest.raises(FileNotFoundError, match="partials"):
        prepare.cmd_tile_wsi_reduce(argparse.Namespace(config=str(config_path)))
