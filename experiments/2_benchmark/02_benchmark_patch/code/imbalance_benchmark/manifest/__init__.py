from __future__ import annotations

import logging
import time
from typing import Any

__all__ = ["log_every"]


def log_every(
    last_logged: float,
    logger: logging.Logger,
    message: str,
    *args: Any,
    interval: float = 30.0,
) -> float:
    """Emit a throttled progress log at most once per ``interval`` seconds.

    Returns the timestamp to pass back in on the next call - unchanged when the
    call was skipped, refreshed when it logged.
    """
    now = time.perf_counter()
    if now - last_logged <= interval:
        return last_logged
    logger.info(message, *args)
    return now
