from __future__ import annotations

from functools import lru_cache

import torch

from imbalance_benchmark.datasets.feature_provenance import (
    load_stored_feature_tensor,
)

__all__ = [
    "load_feature_row",
    "load_slide_features",
    "feature_rows",
    "bank_index",
    "reset_feature_bank",
]


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


def _grow_bank(new_rows: list[torch.Tensor]) -> None:
    """Append newly materialized rows to the bank, promoting dtype on mismatch."""
    global _BANK
    if not new_rows:
        return
    addition = torch.stack(new_rows)
    if _BANK is not None and _BANK.dtype != addition.dtype:
        _BANK, addition = _BANK.float(), addition.float()
    _BANK = addition if _BANK is None else torch.cat([_BANK, addition], dim=0)


def feature_rows(paths: list[str], indices: list[int | None]) -> torch.Tensor:
    """Return a LongTensor of bank row ids, materializing any missing rows.

    ponytail: grows the bank via ``torch.cat`` on every miss batch, which is
    O(n^2) in the number of *grow events* (not rows). In practice there is
    exactly one grow event: ``_tuning_inputs`` loads the full validation
    manifest first, sealing those rows before any per-condition training
    manifest asks for its (already-seen) subset. Revisit only if profiling
    shows repeated grows.
    """
    ids = torch.empty(len(paths), dtype=torch.long)
    new_rows: list[torch.Tensor] = []
    for position, (path, index) in enumerate(zip(paths, indices)):
        key_index, row = _resolve_row(path, index)
        key = (path, key_index)
        if key not in _ROWS:
            _ROWS[key] = len(_ROWS)
            new_rows.append(row)
        ids[position] = _ROWS[key]
    _grow_bank(new_rows)
    return ids


def bank_index(rows: torch.Tensor) -> torch.Tensor:
    """Gather bank rows by id, upcasting to float32 for the model forward pass."""
    if _BANK is None:
        raise RuntimeError("Feature bank is empty; call feature_rows() first")
    return _BANK[rows].float()


def reset_feature_bank() -> None:
    """Drop the bank and its row index.

    A real process builds exactly one bank per run, so this exists for test
    isolation (distinct tests use distinct feature dims, which the bank
    cannot mix once seeded).
    """
    global _BANK
    _BANK = None
    _ROWS.clear()
