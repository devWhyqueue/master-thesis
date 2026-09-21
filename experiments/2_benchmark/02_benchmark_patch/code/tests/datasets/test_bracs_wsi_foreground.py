from __future__ import annotations

import numpy as np

from imbalance_benchmark.datasets.bracs import wsi_foreground as fg


class _CoarseFakeInfo:
    level_downsamples = (1.0, 8.0)
    level_dimensions = ((768, 768), (96, 96))


class _CoarseFakeReader:
    info = _CoarseFakeInfo()

    def read_rect(
        self,
        _location: tuple[int, int],
        size: tuple[int, int],
        resolution: float = 0,
        units: str = "level",
    ) -> np.ndarray:
        assert units == "level"
        assert resolution == 1
        pixels = np.full((size[1], size[0]), 220, dtype=np.uint8)  # background
        pixels[size[1] // 2 :, size[0] // 2 :] = 30  # bottom-right quadrant: tissue
        return np.stack([pixels] * 3, axis=-1)


def test_coarse_tissue_mask_flags_only_the_tissue_quadrant() -> None:
    mask, coarse_downsample = fg.coarse_tissue_mask(_CoarseFakeReader())

    assert coarse_downsample == 8.0
    assert not fg.is_candidate(mask, coarse_downsample, row=0, col=0, downsample=1.0)
    assert fg.is_candidate(mask, coarse_downsample, row=2, col=2, downsample=1.0)


def test_coarse_tissue_mask_is_skipped_for_single_level_slides() -> None:
    class _SingleLevelInfo:
        level_downsamples = (1.0,)

    class _SingleLevelReader:
        info = _SingleLevelInfo()

    mask, coarse_downsample = fg.coarse_tissue_mask(_SingleLevelReader())

    assert mask is None
    assert fg.is_candidate(mask, coarse_downsample, row=0, col=0, downsample=1.0)
