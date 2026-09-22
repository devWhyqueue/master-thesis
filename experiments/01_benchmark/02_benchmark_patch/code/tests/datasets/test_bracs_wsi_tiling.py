from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets.bracs import wsi_tiling
from imbalance_benchmark.datasets.bracs.audit import _rule_mask, validate_tile_manifest
from imbalance_benchmark.datasets.bracs.wsi_foreground import TILE_SIZE

# 3x3 grid: a 2x2 tissue block (each cell has >=2 tissue neighbours) plus one
# isolated tissue cell at (2, 2) that the tissue_neighbors>=2 rule must drop.
_GRID = {
    (0, 0): "tissue", (0, 1): "tissue", (0, 2): "background",
    (1, 0): "tissue", (1, 1): "tissue", (1, 2): "background",
    (2, 0): "background", (2, 1): "background", (2, 2): "tissue",
}


def _cell_pixels(kind: str, size: tuple[int, int]) -> np.ndarray:
    if kind == "tissue":
        gray = np.random.default_rng(0).integers(
            0, 255, size=(size[1], size[0]), dtype=np.uint8
        )
    else:
        gray = np.full((size[1], size[0]), 220, dtype=np.uint8)
    return np.stack([gray] * 3, axis=-1)


class _FakeInfo:
    objective_power = 20.0
    slide_dimensions = (3 * TILE_SIZE, 3 * TILE_SIZE)
    level_downsamples = (1.0,)
    level_dimensions = ((3 * TILE_SIZE, 3 * TILE_SIZE),)


class _FakeReader:
    info = _FakeInfo()

    def read_rect(
        self,
        location: tuple[int, int],
        size: tuple[int, int],
        resolution: float = 0,
        units: str = "level",
    ) -> np.ndarray:
        assert units == "power"
        col, row = location[0] // TILE_SIZE, location[1] // TILE_SIZE
        return _cell_pixels(_GRID[(row, col)], size)


def test_bracs_wsi_tiling_audits_tiles_and_drops_isolated_tissue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(wsi_tiling.WSIReader, "open", lambda _path, **_kw: _FakeReader())

    frame = wsi_tiling.tile_slide(tmp_path / "BRACS_1.svs", "BRACS_1", tmp_path)

    kept = set(
        zip(
            (frame["y"] // TILE_SIZE).tolist(),
            (frame["x"] // TILE_SIZE).tolist(),
        )
    )
    assert kept == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert (frame["tissue_neighbors"] >= 2).all()

    validate_tile_manifest(frame, expected_slides=1)
    assert bool(_rule_mask(frame).all())
    for image_path, expected_hash in frame[["image_path", "sha256"]].itertuples(
        index=False, name=None
    ):
        path = Path(str(image_path))
        assert path.is_file()
        assert compute_sha256(path) == expected_hash


def test_bracs_wsi_tiling_raises_when_no_tile_passes_the_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _BlankReader(_FakeReader):
        def read_rect(
            self,
            _location: tuple[int, int],
            size: tuple[int, int],
            resolution: float = 0,
            units: str = "level",
        ) -> np.ndarray:
            return _cell_pixels("background", size)

    monkeypatch.setattr(wsi_tiling.WSIReader, "open", lambda _path, **_kw: _BlankReader())

    with pytest.raises(ValueError, match="no tiles passed"):
        wsi_tiling.tile_slide(tmp_path / "BRACS_2.svs", "BRACS_2", tmp_path)
