from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import os

import torch

from imbalance_benchmark.datasets.feature_provenance import (
    load_stored_feature_tensor,
)

__all__ = [
    "load_feature_row",
    "load_slide_features",
    "feature_rows",
    "bank_index",
    "bank_is_cpu",
    "reset_feature_bank",
]

# Fraction of free accelerator memory the bank may claim in "auto" placement.
_BANK_DEVICE_FRACTION = 0.75


@lru_cache(maxsize=8192)
def load_slide_features(path: str) -> torch.Tensor:
    """Load a feature tensor and normalize to float (n_instances, dim)."""
    return load_stored_feature_tensor(path).float()


def load_feature_row(path: str, index: int | None = None) -> torch.Tensor:
    """Load one feature vector; a multi-row tensor requires an explicit index."""
    features = load_slide_features(path)
    if index is not None:
        return features[int(index)].squeeze()
    if features.shape[0] == 1:
        return features[0].squeeze()
    raise ValueError(
        f"Feature file {path} has {features.shape[0]} rows; "
        "provide feature_index for multi-row tensors."
    )


# Process-global row bank keyed by (feature_path, normalized index). Reads each
# feature file once regardless of how many ImbalanceDataset instances (train,
# val, per-condition) reference the same patches, and preserves the on-disk
# dtype rather than upcasting eagerly like ``load_slide_features`` does.
_BANK: torch.Tensor | None = None
_ROWS: dict[tuple[str, int], int] = {}
_BANK_CAPACITY = 0


def _resolve_row(path: str, index: int | None) -> tuple[int, torch.Tensor]:
    """Return a normalized row index and the raw (dtype-preserving) row tensor."""
    tensor = load_stored_feature_tensor(path)
    if index is not None:
        return int(index), tensor[int(index)]
    if tensor.shape[0] == 1:
        return 0, tensor[0]
    raise ValueError(
        f"Feature file {path} has {tensor.shape[0]} rows; "
        "provide feature_index for multi-row tensors."
    )


def _target_bank_device(env: Mapping[str, str], num_bytes: int) -> torch.device:
    """Resolve the bank's placement per IMB_FEATURE_BANK_DEVICE (auto|cpu|cuda)."""
    mode = env.get("IMB_FEATURE_BANK_DEVICE", "auto")
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free_bytes, _ = torch.cuda.mem_get_info()
    fits = num_bytes <= _BANK_DEVICE_FRACTION * free_bytes
    return torch.device("cuda") if fits else torch.device("cpu")


def _reserve_bank(rows: list[torch.Tensor], capacity: int) -> None:
    """Allocate final bank once and fill its first free rows."""
    global _BANK, _BANK_CAPACITY
    if not rows:
        return
    if _BANK is None:
        first = rows[0]
        _BANK_CAPACITY = capacity
        _BANK = torch.empty(
            (capacity, *first.shape),
            dtype=first.dtype,
            device=_target_bank_device(
                os.environ, capacity * first.numel() * first.element_size()
            ),
        )
    if len(_ROWS) > _BANK_CAPACITY:
        raise RuntimeError(
            "Feature bank capacity hint is smaller than its unique rows."
        )
    if any(row.dtype != _BANK.dtype or row.shape != _BANK.shape[1:] for row in rows):
        raise ValueError("Feature bank rows must share dtype and shape.")
    start = len(_ROWS) - len(rows)
    _BANK[start : start + len(rows)].copy_(torch.stack(rows), non_blocking=True)


def feature_rows(
    paths: list[str], indices: list[int | None], capacity_hint: int | None = None
) -> torch.Tensor:
    """Return bank row ids, reserving split manifests' full row capacity once."""
    ids = torch.empty(len(paths), dtype=torch.long)
    new_rows: list[torch.Tensor] = []
    for position, (path, index) in enumerate(zip(paths, indices)):
        key_index, row = _resolve_row(path, index)
        key = (path, key_index)
        if key not in _ROWS:
            _ROWS[key] = len(_ROWS)
            new_rows.append(row)
        ids[position] = _ROWS[key]
    # One process loads several manifests into the same bank -- three splits'
    # validation frames, then far smaller per-condition frames. Each hint sizes
    # only its own manifest, so the reservation takes the largest seen rather
    # than letting a later, smaller frame undercut the rows already banked.
    _reserve_bank(new_rows, max(capacity_hint or 0, len(_ROWS)))
    return ids


def bank_index(rows: torch.Tensor) -> torch.Tensor:
    """Gather bank rows by id, upcasting to float32 for the model forward pass."""
    if _BANK is None:
        raise RuntimeError("Feature bank is empty; call feature_rows() first")
    return _BANK[rows.to(_BANK.device)].float()


def bank_is_cpu() -> bool:
    """True when the feature bank is absent or resident on CPU."""
    return _BANK is None or _BANK.device.type == "cpu"


def reset_feature_bank() -> None:
    """Drop the bank and its row index.

    A real process builds exactly one bank per run, so this exists for test
    isolation (distinct tests use distinct feature dims, which the bank
    cannot mix once seeded).
    """
    global _BANK, _BANK_CAPACITY
    _BANK = None
    _BANK_CAPACITY = 0
    _ROWS.clear()
