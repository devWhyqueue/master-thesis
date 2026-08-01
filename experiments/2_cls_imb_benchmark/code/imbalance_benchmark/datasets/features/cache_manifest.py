"""Atomic feature-cache records for serial and multi-GPU extraction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import torch

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets.features.cache import load_slide_features


def cache_records(feature_root: Path) -> dict[str, dict[str, object]]:
    """Load the parent feature-cache manifest, if it exists."""
    path = feature_root / "feature_cache_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_tensor_atomic(tensor: torch.Tensor, path: Path) -> None:
    """Publish a completed tensor only after its complete temporary write."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def record_pending_slide(
    feature_root: Path,
    slide_id: str,
    slide_path: Path,
    ordered_patch_identity: list[str],
    row_count: int,
) -> None:
    """Atomically persist one worker's completed-slide provenance."""
    _write_json_atomic(
        _pending_record_path(feature_root, slide_id),
        {
            "slide_id": slide_id,
            "row_count": row_count,
            "patch_order_sha256": _order_hash(ordered_patch_identity),
            "tensor_sha256": compute_sha256(slide_path),
        },
    )


def merge_pending_slides(
    feature_root: Path, expected: dict[str, tuple[Path, list[str]]]
) -> None:
    """Validate completed worker records then merge them in slide-id order."""
    records = cache_records(feature_root)
    pending = [
        slide_id
        for slide_id in sorted(expected)
        if _pending_exists(feature_root, slide_id)
    ]
    if not pending:
        return
    for slide_id in pending:
        path, identities = expected[slide_id]
        record = _read_pending_record(feature_root, slide_id)
        if record.get("slide_id") != slide_id:
            raise ValueError(
                f"Pending feature record belongs to another slide: {slide_id}"
            )
        _validate_record(slide_id, path, identities, record)
        records[slide_id] = {
            key: value for key, value in record.items() if key != "slide_id"
        }
    _write_json_atomic(feature_root / "feature_cache_manifest.json", records)
    for slide_id in pending:
        _pending_record_path(feature_root, slide_id).unlink()


def validate_cached_slide(
    feature_root: Path,
    slide_id: str,
    slide_path: Path,
    ordered_patch_identity: list[str],
    row_count: int,
) -> None:
    """Refuse reuse when cached rows, patch order, or tensor content changed."""
    record = cache_records(feature_root).get(slide_id)
    if record is None:
        raise ValueError(f"Cached slide {slide_id} lacks row/order/hash provenance")
    _validate_record(slide_id, slide_path, ordered_patch_identity, record, row_count)


def cached_slide_ids(feature_root: Path) -> set[str]:
    """Return slide IDs already committed to the parent cache manifest."""
    return set(cache_records(feature_root))


def _validate_record(
    slide_id: str,
    slide_path: Path,
    ordered_patch_identity: list[str],
    record: dict[str, object],
    row_count: int | None = None,
) -> None:
    rows = len(load_slide_features(str(slide_path))) if row_count is None else row_count
    if record.get("patch_order_sha256") != _order_hash(ordered_patch_identity):
        raise ValueError(f"Cached slide {slide_id} patch order differs")
    if (
        rows != len(ordered_patch_identity)
        or int(str(record.get("row_count", -1))) != rows
    ):
        raise ValueError(f"Cached slide {slide_id} row count differs")
    if record.get("tensor_sha256") != compute_sha256(slide_path):
        raise ValueError(f"Cached slide {slide_id} tensor hash differs")


def _pending_record_path(feature_root: Path, slide_id: str) -> Path:
    digest = hashlib.sha256(slide_id.encode("utf-8")).hexdigest()
    return feature_root / ".feature_cache_pending" / f"{digest}.json"


def _pending_exists(feature_root: Path, slide_id: str) -> bool:
    return _pending_record_path(feature_root, slide_id).is_file()


def _read_pending_record(feature_root: Path, slide_id: str) -> dict[str, object]:
    return json.loads(
        _pending_record_path(feature_root, slide_id).read_text(encoding="utf-8")
    )


def _order_hash(ordered_patch_identity: list[str]) -> str:
    payload = json.dumps(ordered_patch_identity, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(
    path: Path, payload: dict[str, object] | dict[str, dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
