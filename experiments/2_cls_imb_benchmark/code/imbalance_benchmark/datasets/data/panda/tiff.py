"""PANDA TIFF reader helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import openslide
from PIL import Image


@contextmanager
def official_tiff_guard() -> Iterator[None]:
    """Allow trusted official PANDA TIFFs beyond Pillow's generic limit."""
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous


@contextmanager
def source_reader(path: str) -> Iterator[Image.Image | openslide.OpenSlide]:
    """Open a WSI through OpenSlide, with a Pillow test-data fallback."""
    try:
        source = openslide.OpenSlide(path)
    except openslide.OpenSlideUnsupportedFormatError:
        with Image.open(path) as source:
            yield source
    else:
        try:
            yield source
        finally:
            source.close()
