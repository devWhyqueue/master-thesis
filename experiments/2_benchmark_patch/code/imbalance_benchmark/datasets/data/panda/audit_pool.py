"""Crash-isolated ProcessPoolExecutor runner for one shard's audit jobs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

AuditJob = tuple[int, int, pd.Series, dict[str, Any]]
AuditJobFn = Callable[[AuditJob], pd.DataFrame]


def run_audit_jobs(
    jobs: list[AuditJob], workers: int, run_job: AuditJobFn
) -> tuple[list[pd.DataFrame], list[str]]:
    """Run every job concurrently; isolate and exclude any slide whose worker crashes.

    A native crash (segfault) in OpenSlide/libtiff/libjpeg kills its worker
    process and breaks every other future still pending in the same pool,
    even ones unrelated to the crash. Recovering the futures that already
    finished and retrying only the interrupted ones alone (one worker each)
    narrows a pool-wide failure down to the actual offending slide(s)
    instead of losing the whole shard's progress.
    """
    audited, interrupted = _submit_jobs(jobs, workers, run_job)
    crashed: list[str] = []
    for job in interrupted:
        result = _run_job_isolated(job, run_job)
        if result is None:
            crashed.append(str(job[2].slide_id))
        else:
            audited.append(result)
    return audited, crashed


def _submit_jobs(
    jobs: list[AuditJob], workers: int, run_job: AuditJobFn
) -> tuple[list[pd.DataFrame], list[AuditJob]]:
    audited: list[pd.DataFrame] = []
    interrupted: list[AuditJob] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future, job in futures.items():
            try:
                audited.append(future.result())
            except BrokenProcessPool:
                interrupted.append(job)
    return audited, interrupted


def _run_job_isolated(job: AuditJob, run_job: AuditJobFn) -> pd.DataFrame | None:
    """Retry one job alone; return None if its worker crashes again."""
    with ProcessPoolExecutor(max_workers=1) as pool:
        try:
            return pool.submit(run_job, job).result()
        except BrokenProcessPool:
            logger.error(
                "PANDA audit worker crashed auditing slide %s (native crash, not "
                "a Python exception); excluding it from this shard",
                job[2].slide_id,
            )
            return None
