"""Training progress helpers without experiment-specific I/O."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_training_progress(
    result_dir: Path, payload: dict[str, Any], loss: float | None = None
) -> None:
    if loss is not None:
        payload["loss"] = loss
    progress_path = result_dir / "progress.json"
    progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def progress_payload(
    method: str,
    seed: int,
    device: object,
    status: str,
    epoch: int,
    epochs: int,
) -> dict[str, object]:
    return {
        "method": method,
        "seed": seed,
        "device": str(device),
        "status": status,
        "epoch": epoch,
        "epochs": epochs,
        "epoch_fraction": epoch / max(epochs, 1),
    }


_write_training_progress = write_training_progress
_progress_payload = progress_payload
