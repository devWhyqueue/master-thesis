from __future__ import annotations

from code.metadata import WSI_METHOD_METADATA

BAG_METHODS = set(WSI_METHOD_METADATA)


def method_metadata(method: str) -> dict[str, object]:
    """Return metadata for one WSI-bag benchmark method."""
    return WSI_METHOD_METADATA[method]
