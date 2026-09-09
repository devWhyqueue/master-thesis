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
    """Sort candidate top entries by (-similarity, bank_index)."""
    sims_cpu, idx_cpu = sims.cpu().numpy(), indices.cpu().numpy()
    n_queries = sims_cpu.shape[0]
    out_sims = np.zeros((n_queries, top), dtype=np.float32)
    out_idx = np.zeros((n_queries, top), dtype=np.int64)

    for q in range(n_queries):
        order = np.lexsort((idx_cpu[q], -sims_cpu[q]))[:top]
        out_sims[q], out_idx[q] = sims_cpu[q, order], idx_cpu[q, order]

    return torch.from_numpy(out_sims), torch.from_numpy(out_idx)


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


def top_neighbours(
    query_features: torch.Tensor,
    bank_features: torch.Tensor,
    top: int = NEIGHBOUR_TOP,
    query_batch_size: int = 1024,
    bank_batch_size: int = 4096,
    window_factor: int = 128,
) -> tuple[np.ndarray, np.ndarray, SearchStats]:
    """Compute exact cosine top-k neighbours via batching without full matrix."""
    start = time.perf_counter()
    q_norm, b_norm = _normalize(query_features), _normalize(bank_features)
    cand_k = min(b_norm.shape[0], max(top, window_factor))
    out_i, out_s = _gather_query_chunks(
        q_norm, b_norm, top, cand_k, query_batch_size, bank_batch_size
    )
    dev = query_features.device.type
    mem = (
        torch.cuda.max_memory_allocated(query_features.device) / (1024 * 1024)
        if dev == "cuda" and torch.cuda.is_available()
        else 0.0
    )
    return (
        out_i,
        out_s,
        SearchStats(
            time.perf_counter() - start, mem, dev, query_batch_size, bank_batch_size
        ),
    )


def vote(
    neighbour_labels: np.ndarray, k: int, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Plurality vote among first k neighbours with first-max tie resolution."""
    sub = neighbour_labels[:, :k]
    n_samples = sub.shape[0]
    preds = np.zeros(n_samples, dtype=np.int64)
    probs = np.zeros((n_samples, n_classes), dtype=np.float64)

    for i in range(n_samples):
        counts = np.bincount(sub[i], minlength=n_classes)
        probs[i] = counts / float(k)
        # np.argmax returns first occurrence of max value -> lowest class index
        preds[i] = np.argmax(counts)

    return preds, probs
