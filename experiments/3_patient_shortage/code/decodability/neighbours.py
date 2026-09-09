"""Exact cosine k-NN search and fixed-order voting."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from decodability import NEIGHBOUR_TOP

__all__ = ["SearchStats", "top_neighbours", "vote"]


@dataclass(frozen=True)
class SearchStats:
    """Resource and timing metadata for k-NN search."""

    wall_time_s: float
    peak_memory_mb: float
    device: str
    query_batch_size: int
    bank_batch_size: int


def _normalize(mat: torch.Tensor) -> torch.Tensor:
    """Normalize rows to unit L2 norm."""
    norms = torch.norm(mat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return mat / norms


def _rank_chunk(
    sims: torch.Tensor,
    indices: torch.Tensor,
    top: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort candidate top entries by (-similarity, bank_index), per row."""
    sims_cpu, idx_cpu = sims.cpu().numpy(), indices.cpu().numpy()
    n_queries, n_cols = sims_cpu.shape
    flat_sims, flat_idx = sims_cpu.ravel(), idx_cpu.ravel()
    row_key = np.repeat(np.arange(n_queries), n_cols)
    order = np.lexsort((flat_idx, -flat_sims, row_key))
    top_pos = order.reshape(n_queries, n_cols)[:, :top]

    return torch.from_numpy(flat_sims[top_pos]), torch.from_numpy(flat_idx[top_pos])


def _scan_bank_for_query_chunk(
    q_chunk: torch.Tensor, b_norm: torch.Tensor, candidate_k: int, bank_batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan normalized bank in chunks and maintain running top candidates."""
    best_sims: torch.Tensor | None = None
    best_idx: torch.Tensor | None = None
    for b_start in range(0, b_norm.shape[0], bank_batch):
        b_chunk = b_norm[b_start : b_start + bank_batch]
        sims = torch.matmul(q_chunk, b_chunk.T)
        k_val = min(candidate_k, b_chunk.shape[0])
        vals, locs = torch.topk(sims, k=k_val, dim=1, largest=True)
        g_locs = locs + b_start
        if best_sims is None or best_idx is None:
            best_sims, best_idx = vals, g_locs
        else:
            cat_s = torch.cat([best_sims, vals], dim=1)
            cat_i = torch.cat([best_idx, g_locs], dim=1)
            best_sims, best_idx = _rank_chunk(
                cat_s, cat_i, min(candidate_k, cat_s.shape[1])
            )
            if q_chunk.device.type == "cuda":
                best_sims, best_idx = (
                    best_sims.to(q_chunk.device),
                    best_idx.to(q_chunk.device),
                )
    assert best_sims is not None and best_idx is not None
    return best_sims, best_idx


def _gather_query_chunks(
    q_norm: torch.Tensor,
    b_norm: torch.Tensor,
    top: int,
    cand_k: int,
    q_batch: int,
    b_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate query chunks, rank top-k, and concatenate output arrays."""
    all_i, all_s = [], []
    for q_start in range(0, q_norm.shape[0], q_batch):
        q_chunk = q_norm[q_start : q_start + q_batch]
        b_sims, b_idx = _scan_bank_for_query_chunk(q_chunk, b_norm, cand_k, b_batch)
        f_sims, f_idx = _rank_chunk(b_sims, b_idx, min(top, b_sims.shape[1]))
        all_s.append(f_sims.numpy())
        all_i.append(f_idx.numpy())
    out_i = np.vstack(all_i) if all_i else np.zeros((0, top), dtype=np.int64)
    out_s = np.vstack(all_s) if all_s else np.zeros((0, top), dtype=np.float32)
    return out_i, out_s


def _peak_memory_mb(device: torch.device, device_type: str) -> float:
    """Return CUDA peak allocated memory in MB, or 0.0 off-GPU."""
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def top_neighbours(
    query_features: torch.Tensor,
    bank_features: torch.Tensor,
    top: int = NEIGHBOUR_TOP,
    query_batch_size: int = 1024,
    bank_batch_size: int | None = None,
    window_factor: int = 128,
) -> tuple[np.ndarray, np.ndarray, SearchStats]:
    """Compute exact cosine top-k neighbours via batching without full matrix.

    ``bank_batch_size`` defaults to the whole bank (at most ~460 MB for this
    experiment's train sets), so the scan is one matmul + one topk per query
    chunk instead of several chunk merges.
    """
    start = time.perf_counter()
    q_norm, b_norm = _normalize(query_features), _normalize(bank_features)
    resolved_bank_batch = bank_batch_size or b_norm.shape[0]
    cand_k = min(b_norm.shape[0], max(top, window_factor))
    out_i, out_s = _gather_query_chunks(
        q_norm, b_norm, top, cand_k, query_batch_size, resolved_bank_batch
    )
    dev = query_features.device.type
    mem = _peak_memory_mb(query_features.device, dev)
    stats = SearchStats(
        time.perf_counter() - start, mem, dev, query_batch_size, resolved_bank_batch
    )
    return out_i, out_s, stats


def vote(
    neighbour_labels: np.ndarray, k: int, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Plurality vote among first k neighbours with first-max tie resolution."""
    sub = neighbour_labels[:, :k]
    n_samples = sub.shape[0]
    row_offsets = np.arange(n_samples)[:, None] * n_classes
    counts = np.bincount(
        (sub + row_offsets).ravel(), minlength=n_samples * n_classes
    ).reshape(n_samples, n_classes)
    probs = counts / float(k)
    # np.argmax returns first occurrence of max value -> lowest class index
    preds = np.argmax(counts, axis=1).astype(np.int64)

    return preds, probs
