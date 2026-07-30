from __future__ import annotations

from collections.abc import Callable
import os
import subprocess

__all__ = ["DEFAULT_QUEUE_CAP", "check_queue_cap"]

DEFAULT_QUEUE_CAP = 25


def _squeue_count() -> int:
    """Count this user's queued and running SLURM tasks via ``squeue``."""
    user = os.environ.get("USER") or os.environ.get("USERNAME", "")
    result = subprocess.run(
        ["squeue", "-u", user, "-r", "-h"],
        capture_output=True,
        text=True,
        check=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def check_queue_cap(
    cap: int = DEFAULT_QUEUE_CAP, count: Callable[[], int] = _squeue_count
) -> None:
    """Refuse to submit another job once queued+running is at or over the cap.

    Adaptive tuning rounds submit themselves from a running job rather than
    all at once up front, so this is the guard against ever exceeding the
    cluster's queue budget - never a static property of the workflow DAG.
    """
    current = count()
    if current >= cap:
        raise RuntimeError(
            f"Queue at {current} tasks (cap {cap}); refusing to submit another job. "
            "Wait for jobs to clear before resubmitting."
        )
