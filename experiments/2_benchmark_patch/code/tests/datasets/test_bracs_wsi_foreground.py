from __future__ import annotations

import numpy as np
from PIL import Image

from imbalance_benchmark.datasets.bracs import wsi_foreground as fg


class _CoarseFakeSlide:
    level_downsamples = (1.0, 8.0)
    level_dimensions = ((768, 768), (96, 96))

    def read_region(
        self, _location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        assert level == 1
        pixels = np.full((size[1], size[0]), 220, dtype=np.uint8)  # background
        pixels[size[1] // 2 :, size[0] // 2 :] = 30  # bottom-right quadrant: tissue
        return Image.fromarray(pixels, mode="L").convert("RGB")


def test_coarse_tissue_mask_flags_only_the_tissue_quadrant() -> None:
    mask, coarse_downsample = fg.coarse_tissue_mask(_CoarseFakeSlide())

    assert coarse_downsample == 8.0
    assert not fg.is_candidate(mask, coarse_downsample, row=0, col=0, downsample=1.0)
    assert fg.is_candidate(mask, coarse_downsample, row=2, col=2, downsample=1.0)


def test_coarse_tissue_mask_is_skipped_for_single_level_slides() -> None:
    class _SingleLevelSlide:
        level_downsamples = (1.0,)

    mask, coarse_downsample = fg.coarse_tissue_mask(_SingleLevelSlide())

    assert mask is None
    assert fg.is_candidate(mask, coarse_downsample, row=0, col=0, downsample=1.0)
