from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common import write_progress

logger = logging.getLogger(__name__)


def _slice_split_rows(
    frame: pd.DataFrame, max_train: int | None, max_eval: int | None
) -> pd.DataFrame:
    parts = []
    split_limits = [("train", max_train), ("val", max_eval), ("test", max_eval)]
    for split, max_rows in split_limits:
        split_frame = frame[frame["split"] == split]
        if max_rows:
            split_frame = split_frame.groupby("cancer_type", group_keys=False).head(
                int(max_rows)
            )
        parts.append(split_frame)
    return pd.concat(parts, ignore_index=True)


def _write_training_progress(
    result_dir: Path, payload: dict[str, Any], loss: float | None = None
) -> None:
    """Write training progress JSON and log line."""
    if loss is not None:
        payload["loss"] = loss
    write_progress(result_dir / "progress.json", payload)
    logger.info(
        "progress method=%s seed=%s status=%s epoch=%s/%s device=%s loss=%s",
        payload["method"],
        payload["seed"],
        payload["status"],
        payload.get("epoch", "NA"),
        payload.get("epochs", "NA"),
        payload.get("device", "NA"),
        f"{loss:.4f}" if loss is not None else "NA",
    )


def _progress_payload(
    method: str,
    seed: int,
    device: object,
    status: str,
    epoch: int,
    epochs: int,
) -> dict[str, object]:
    """Build a standard training-progress payload."""
    return {
        "method": method,
        "seed": seed,
        "device": str(device),
        "status": status,
        "epoch": epoch,
        "epochs": epochs,
        "epoch_fraction": epoch / max(epochs, 1),
    }
