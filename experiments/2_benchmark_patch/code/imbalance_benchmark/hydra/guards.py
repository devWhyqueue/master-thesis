from __future__ import annotations

from collections.abc import Callable
import os
import subprocess
import time

__all__ = ["DEFAULT_QUEUE_CAP", "check_queue_cap", "DeadlineGuard"]

DEFAULT_QUEUE_CAP = 100


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


class DeadlineGuard:
    """Stop a resumable work loop before SLURM kills the task mid-item.

    Converts a TIMEOUT (task FAILED, the in-flight item's work discarded)
    into a clean partial completion (the task exits 0 before the item that
    would overrun, and resume -- already item-granular -- picks up the
    untouched remainder).

    Absent ``SLURM_JOB_END_TIME`` (no allocation, or a local/test run), the
    guard never fires: :meth:`should_stop` always returns ``False``.
    """

    def __init__(self, seed_margin_seconds: float) -> None:
        end = os.environ.get("SLURM_JOB_END_TIME")
        self._end_time = float(end) if end else None
        self._margin = seed_margin_seconds
        self._item_started: float | None = None

    def should_stop(self) -> bool:
        """True when starting one more item risks not finishing before the deadline."""
        if self._end_time is None:
            return False
        return (self._end_time - time.time()) < self._margin

    def start_item(self) -> None:
        """Mark the next item as started, timing it for the next margin update."""
        self._item_started = time.perf_counter()

    def finish_item(self) -> None:
        """Widen the margin to the largest item duration actually observed.

        # ponytail: a running max is deliberately naive -- a
        # per-(condition, method) estimate is the upgrade path if this
        # margin turns out to discard real wall time on every wave.
        """
        if self._item_started is None:
            return
        self._margin = max(self._margin, time.perf_counter() - self._item_started)
        self._item_started = None
