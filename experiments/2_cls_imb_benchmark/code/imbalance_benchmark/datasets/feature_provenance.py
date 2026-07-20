from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from imbalance_benchmark.common import compute_sha256

FEATURE_DIM = 2560
VIRCHOW2_MODEL = "hf-hub:paige-ai/Virchow2"
VIRCHOW2_REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
VIRCHOW2_WEIGHTS_SHA256 = (
    "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
)


def resolve_feature_provenance(config: dict[str, object]) -> dict[str, object]:
    """Resolve and validate the sole encoder/projection permitted by the protocol."""
    model_name = str(config.get("model_name", VIRCHOW2_MODEL))
    if model_name != VIRCHOW2_MODEL:
        raise ValueError(f"The benchmark requires frozen Virchow2, not {model_name!r}")
    dtype = str(config.get("dtype", "float16"))
    if dtype not in {"float16", "float32"}:
        raise ValueError("feature_extraction.dtype must be float16 or float32")
    revision = str(config.get("revision", VIRCHOW2_REVISION))
    weights_sha256 = str(config.get("weights_sha256", VIRCHOW2_WEIGHTS_SHA256))
    if revision != VIRCHOW2_REVISION or weights_sha256 != VIRCHOW2_WEIGHTS_SHA256:
        raise ValueError(
            "Virchow2 revision and weights hash must match the frozen encoder"
        )
    return {
        "encoder_id": VIRCHOW2_MODEL,
        "encoder_revision": revision,
        "weights_sha256": weights_sha256,
        "pooling": "cls_plus_mean_patch_tokens",
        "feature_dim": FEATURE_DIM,
        "dtype": dtype,
    }


def validate_feature_cache(feature_root: Path, provenance: dict[str, object]) -> None:
    """Write or verify immutable encoder metadata before reusing cached features."""
    metadata_path = feature_root / "feature_provenance.json"
    if metadata_path.exists():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if recorded != provenance:
            raise ValueError(
                "Cached feature provenance differs from the requested encoder"
            )
        return
    if any(feature_root.glob("*.pt")):
        raise ValueError("Cached features lack feature provenance metadata")
    metadata_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def resolve_feature_snapshot(config: dict[str, object]) -> Path:
    """Download the pinned encoder snapshot and verify its published weight hash."""
    provenance = resolve_feature_provenance(config)
    snapshot = Path(
        snapshot_download(
            "paige-ai/Virchow2",
            revision=str(provenance["encoder_revision"]),
            allow_patterns=["config.json", "model.safetensors"],
        )
    )
    weights = snapshot / "model.safetensors"
    if not weights.is_file() or compute_sha256(weights) != provenance["weights_sha256"]:
        raise ValueError("Downloaded Virchow2 weights do not match the frozen hash")
    return snapshot


def patch_sort_key(item: str) -> tuple[int, int]:
    """Sort patch identifiers by region and patch index."""
    region, index = item.split("_")[:2]
    return int(region), int(index)


def _order_hash(ordered_patch_identity: list[str]) -> str:
    payload = json.dumps(ordered_patch_identity, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_records(feature_root: Path) -> dict[str, dict[str, object]]:
    path = feature_root / "feature_cache_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record_cached_slide(
    feature_root: Path,
    slide_id: str,
    slide_path: Path,
    ordered_patch_identity: list[str],
    row_count: int,
) -> None:
    """Record row, order, and tensor hashes for one newly extracted slide."""
    records = _cache_records(feature_root)
    records[slide_id] = {
        "row_count": row_count,
        "patch_order_sha256": _order_hash(ordered_patch_identity),
        "tensor_sha256": compute_sha256(slide_path),
    }
    (feature_root / "feature_cache_manifest.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_cached_slide(
    feature_root: Path,
    slide_id: str,
    slide_path: Path,
    ordered_patch_identity: list[str],
    row_count: int,
) -> None:
    """Refuse reuse when cached rows, patch order, or tensor content changed."""
    record = _cache_records(feature_root).get(slide_id)
    if record is None:
        raise ValueError(f"Cached slide {slide_id} lacks row/order/hash provenance")
    if record.get("patch_order_sha256") != _order_hash(ordered_patch_identity):
        raise ValueError(f"Cached slide {slide_id} patch order differs")
    if (
        row_count != len(ordered_patch_identity)
        or int(str(record.get("row_count", -1))) != row_count
    ):
        raise ValueError(f"Cached slide {slide_id} row count differs")
    if record.get("tensor_sha256") != compute_sha256(slide_path):
        raise ValueError(f"Cached slide {slide_id} tensor hash differs")
