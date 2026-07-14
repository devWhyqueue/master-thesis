from __future__ import annotations

import json
from pathlib import Path

FEATURE_DIM = 2560
VIRCHOW2_MODEL = "hf-hub:paige-ai/Virchow2"


def resolve_feature_provenance(config: dict[str, object]) -> dict[str, object]:
    """Resolve and validate the sole encoder/projection permitted by the protocol."""
    model_name = str(config.get("model_name", VIRCHOW2_MODEL))
    if model_name != VIRCHOW2_MODEL:
        raise ValueError(f"The benchmark requires frozen Virchow2, not {model_name!r}")
    dtype = str(config.get("dtype", "float16"))
    if dtype not in {"float16", "float32"}:
        raise ValueError("feature_extraction.dtype must be float16 or float32")
    return {
        "encoder_id": VIRCHOW2_MODEL,
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


def patch_sort_key(item: str) -> tuple[int, int]:
    """Sort patch identifiers by region and patch index."""
    region, index = item.split("_")[:2]
    return int(region), int(index)
