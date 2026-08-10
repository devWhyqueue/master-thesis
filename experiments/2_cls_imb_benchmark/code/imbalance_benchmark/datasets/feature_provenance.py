from __future__ import annotations

import json
import hashlib
import os
from uuid import uuid4
from functools import lru_cache
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from imbalance_benchmark.common import compute_sha256

FEATURE_DIM = 2560
VIRCHOW2_MODEL = "hf-hub:paige-ai/Virchow2"
VIRCHOW2_REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
VIRCHOW2_WEIGHTS_SHA256 = (
    "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
)


@lru_cache(maxsize=512)
def load_stored_feature_tensor(path: str) -> torch.Tensor:
    """Load and normalize a stored tensor without discarding its dtype."""
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(tensor, dict):
        class_token, mean_token = tensor.get("cls"), tensor.get("mean_patch")
        tensor = (
            torch.cat([class_token, mean_token], dim=-1)
            if class_token is not None and mean_token is not None
            else next(value for value in tensor.values() if torch.is_tensor(value))
        )
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim > 2:
        return tensor.reshape(-1, tensor.shape[-1])
    return tensor


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
    temporary = metadata_path.with_suffix(
        metadata_path.suffix + f".{uuid4().hex}.partial"
    )
    temporary.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, metadata_path)


def validate_preextracted_features(
    manifest_path: Path,
    feature_paths: list[Path],
    provenance: dict[str, object],
) -> dict[str, int]:
    """Verify a signed manifest and every pre-extracted tensor it locks."""
    payload = load_signed_feature_provenance(manifest_path)
    if payload.get("provenance") != provenance:
        raise ValueError("Pre-extracted features do not use pinned Virchow2 provenance")
    records = payload.get("chunks")
    if not isinstance(records, dict) or set(records) != {p.name for p in feature_paths}:
        raise ValueError("Provenance chunk inventory differs from feature files")
    row_counts = {}
    seen_patch_ids: set[str] = set()
    for path in feature_paths:
        record = records[path.name]
        if not isinstance(record, dict):
            raise ValueError(f"Invalid provenance record for {path.name}")
        row_counts[str(path)] = _validate_preextracted_tensor(
            path, record, seen_patch_ids
        )
    return row_counts


def load_signed_feature_provenance(path: Path) -> dict[str, object]:
    """Load an intact signed pre-extracted feature manifest."""
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("TCGA-UT requires a signed provenance manifest")
    if sidecar.read_text(encoding="utf-8").strip() != compute_sha256(path):
        raise ValueError("TCGA-UT signed provenance manifest was altered")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("TCGA-UT provenance manifest must be a mapping")
    return payload


def _validate_preextracted_tensor(
    path: Path, record: dict[str, object], seen_patch_ids: set[str]
) -> int:
    if record.get("tensor_sha256") != compute_sha256(path):
        raise ValueError(f"TCGA-UT tensor hash differs for {path.name}")
    tensor = load_stored_feature_tensor(str(path))
    if (
        tensor.ndim != 2
        or tensor.shape[1] != FEATURE_DIM
        or record.get("feature_dim") != FEATURE_DIM
    ):
        raise ValueError(f"TCGA-UT feature dimension differs for {path.name}")
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in {"float16", "float32"} or record.get("dtype") != dtype:
        raise ValueError(f"TCGA-UT tensor dtype differs for {path.name}")
    if record.get("row_count") != tensor.shape[0]:
        raise ValueError(f"TCGA-UT tensor row count differs for {path.name}")
    identities = record.get("ordered_patch_ids")
    if not isinstance(identities, list) or not all(
        isinstance(identity, str) and identity for identity in identities
    ):
        raise ValueError(f"TCGA-UT ordered patch identities missing for {path.name}")
    if len(identities) != tensor.shape[0]:
        raise ValueError(f"TCGA-UT patch identity count differs for {path.name}")
    if record.get("patch_order_sha256") != _order_hash(identities):
        raise ValueError(f"TCGA-UT patch order hash differs for {path.name}")
    duplicates = seen_patch_ids.intersection(identities)
    if duplicates:
        raise ValueError(
            f"TCGA-UT duplicate patch identities: {sorted(duplicates)[:5]}"
        )
    seen_patch_ids.update(identities)
    return tensor.shape[0]


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
