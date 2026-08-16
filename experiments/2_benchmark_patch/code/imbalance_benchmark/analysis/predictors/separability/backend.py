"""Exact CPU and CUDA implementations of streamed RQ3 nearest-neighbour probes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KNNConfig:
    """Block sizes and fixed neighbour count for an exact probe."""

    k: int
    chunk_size: int
    reference_chunk_size: int


@dataclass(frozen=True)
class ProbeData:
    """Feature and label arrays used by one reference-versus-validation probe."""

    ref_x: np.ndarray
    ref_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    n_classes: int


def cuda_available() -> bool:
    """Whether the current process can execute the exact CUDA distance pass."""
    return torch.cuda.is_available()


def _log_chunk(chunk_number: int, n_chunks: int) -> None:
    if chunk_number == 1 or chunk_number % 10 == 0 or chunk_number == n_chunks:
        logger.info("rq3: knn query chunk %d/%d", chunk_number, n_chunks)


def _log_block(
    chunk_number: int, n_chunks: int, ref_start: int, config: KNNConfig, n_ref: int
) -> None:
    logger.info(
        "rq3: knn query chunk %d/%d reference block %d/%d",
        chunk_number,
        n_chunks,
        ref_start // config.reference_chunk_size + 1,
        (n_ref + config.reference_chunk_size - 1) // config.reference_chunk_size,
    )


def _cpu_topk(
    ref_x: np.ndarray,
    ref_sq: np.ndarray,
    chunk: np.ndarray,
    config: KNNConfig,
    log_block: Callable[[int], None],
) -> tuple[np.ndarray, np.ndarray]:
    k = min(config.k, ref_x.shape[0])
    best_d2 = np.full((chunk.shape[0], k), np.inf, dtype=ref_x.dtype)
    best_idx = np.full((chunk.shape[0], k), ref_x.shape[0], dtype=np.intp)
    chunk_sq = (chunk**2).sum(axis=1, keepdims=True)
    for ref_start in range(0, ref_x.shape[0], config.reference_chunk_size):
        ref_end = min(ref_start + config.reference_chunk_size, ref_x.shape[0])
        log_block(ref_start)
        d2 = (
            chunk_sq
            - 2.0 * chunk @ ref_x[ref_start:ref_end].T
            + ref_sq[:, ref_start:ref_end]
        )
        local_k = min(k, ref_end - ref_start)
        local_idx = np.argpartition(d2, local_k - 1, axis=1)[:, :local_k]
        local_d2 = np.take_along_axis(d2, local_idx, axis=1)
        candidate_d2 = np.concatenate((best_d2, local_d2), axis=1)
        candidate_idx = np.concatenate((best_idx, local_idx + ref_start), axis=1)
        selected = np.argpartition(candidate_d2, k - 1, axis=1)[:, :k]
        best_d2 = np.take_along_axis(candidate_d2, selected, axis=1)
        best_idx = np.take_along_axis(candidate_idx, selected, axis=1)
    return best_d2, best_idx


def _gpu_topk(
    ref_x: torch.Tensor,
    ref_sq: torch.Tensor,
    chunk: torch.Tensor,
    config: KNNConfig,
    log_block: Callable[[int], None],
) -> tuple[torch.Tensor, torch.Tensor]:
    k = min(config.k, ref_x.shape[0])
    best_d2 = torch.full(
        (chunk.shape[0], k), float("inf"), device="cuda", dtype=ref_x.dtype
    )
    best_idx = torch.full(
        (chunk.shape[0], k), ref_x.shape[0], device="cuda", dtype=torch.long
    )
    chunk_sq = chunk.square().sum(dim=1, keepdim=True)
    for ref_start in range(0, ref_x.shape[0], config.reference_chunk_size):
        ref_end = min(ref_start + config.reference_chunk_size, ref_x.shape[0])
        log_block(ref_start)
        d2 = (
            chunk_sq
            - 2.0 * chunk @ ref_x[ref_start:ref_end].T
            + ref_sq[ref_start:ref_end]
        )
        local_k = min(k, ref_end - ref_start)
        local_d2, local_idx = torch.topk(
            d2, local_k, dim=1, largest=False, sorted=False
        )
        candidate_d2 = torch.cat((best_d2, local_d2), dim=1)
        candidate_idx = torch.cat((best_idx, local_idx + ref_start), dim=1)
        best_d2, selected = torch.topk(
            candidate_d2, k, dim=1, largest=False, sorted=False
        )
        best_idx = torch.gather(candidate_idx, 1, selected)
    return best_d2, best_idx


def _record(
    outputs: tuple[np.ndarray, np.ndarray],
    data: ProbeData,
    start: int,
    best_d2: np.ndarray,
    best_idx: np.ndarray,
) -> None:
    preds, nn_correct = outputs
    end = start + len(best_idx)
    neighbor_labels = data.ref_y[best_idx]
    preds[start:end] = [
        np.bincount(row, minlength=data.n_classes).argmax() for row in neighbor_labels
    ]
    nearest = best_idx[np.arange(len(best_idx)), best_d2.argmin(axis=1)]
    nn_correct[start:end] = data.ref_y[nearest] == data.val_y[start:end]


def _cpu_probe(data: ProbeData, config: KNNConfig) -> tuple[np.ndarray, np.ndarray]:
    ref_sq = (data.ref_x**2).sum(axis=1)[None, :]
    n_chunks = (len(data.val_x) + config.chunk_size - 1) // config.chunk_size
    outputs = (
        np.empty(len(data.val_x), dtype=data.ref_y.dtype),
        np.empty(len(data.val_x), dtype=bool),
    )
    for chunk_number, start in enumerate(
        range(0, len(data.val_x), config.chunk_size), start=1
    ):
        chunk = data.val_x[start : start + config.chunk_size]
        _log_chunk(chunk_number, n_chunks)
        log_block = lambda ref_start: _log_block(
            chunk_number, n_chunks, ref_start, config, len(data.ref_x)
        )
        best_d2, best_idx = _cpu_topk(data.ref_x, ref_sq, chunk, config, log_block)
        _record(outputs, data, start, best_d2, best_idx)
    return outputs


def _gpu_probe(data: ProbeData, config: KNNConfig) -> tuple[np.ndarray, np.ndarray]:
    n_chunks = (len(data.val_x) + config.chunk_size - 1) // config.chunk_size
    outputs = (
        np.empty(len(data.val_x), dtype=data.ref_y.dtype),
        np.empty(len(data.val_x), dtype=bool),
    )
    with torch.inference_mode():
        ref_x_device = torch.as_tensor(data.ref_x, device="cuda")
        ref_sq = ref_x_device.square().sum(dim=1)
        for chunk_number, start in enumerate(
            range(0, len(data.val_x), config.chunk_size), start=1
        ):
            chunk = torch.as_tensor(
                data.val_x[start : start + config.chunk_size], device="cuda"
            )
            _log_chunk(chunk_number, n_chunks)
            log_block = lambda ref_start: _log_block(
                chunk_number, n_chunks, ref_start, config, len(data.ref_x)
            )
            best_d2, best_idx = _gpu_topk(
                ref_x_device, ref_sq, chunk, config, log_block
            )
            _record(outputs, data, start, best_d2.cpu().numpy(), best_idx.cpu().numpy())
    return outputs


def knn_and_nn_probe(
    data: ProbeData, config: KNNConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Choose CUDA when available, otherwise retain the exact NumPy implementation."""
    if not cuda_available():
        return _cpu_probe(data, config)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _gpu_probe(data, config)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
