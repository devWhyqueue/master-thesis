from __future__ import annotations

import time

import pytest

from imbalance_benchmark.hydra.guards import DeadlineGuard, check_queue_cap


def test_check_queue_cap_passes_when_under_cap():
    check_queue_cap(cap=25, count=lambda: 24)


def test_check_queue_cap_raises_at_cap():
    with pytest.raises(RuntimeError, match="Queue at 25"):
        check_queue_cap(cap=25, count=lambda: 25)


def test_check_queue_cap_raises_over_cap():
    with pytest.raises(RuntimeError, match="cap 25"):
        check_queue_cap(cap=25, count=lambda: 40)


def test_check_queue_cap_uses_the_default_cap_of_100():
    with pytest.raises(RuntimeError, match="cap 100"):
        check_queue_cap(count=lambda: 100)


def test_guard_is_a_no_op_without_slurm_job_end_time(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)

    guard = DeadlineGuard(seed_margin_seconds=1e9)

    assert guard.should_stop() is False


def test_guard_stops_before_an_item_that_would_overrun(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(time.time() + 60))

    guard = DeadlineGuard(seed_margin_seconds=120)

    assert guard.should_stop() is True


def test_guard_allows_an_item_with_ample_margin(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(time.time() + 3600))

    guard = DeadlineGuard(seed_margin_seconds=120)

    assert guard.should_stop() is False


def test_guard_widens_margin_to_the_largest_observed_item(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    guard = DeadlineGuard(seed_margin_seconds=1.0)
    ticks = iter([100.0, 105.0])
    monkeypatch.setattr(
        "imbalance_benchmark.hydra.guards.time.perf_counter", lambda: next(ticks)
    )

    guard.start_item()
    guard.finish_item()

    assert guard._margin == 5.0


def test_guard_never_shrinks_margin_below_a_larger_earlier_item(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    guard = DeadlineGuard(seed_margin_seconds=1.0)
    ticks = iter([0.0, 10.0, 10.0, 11.0])
    monkeypatch.setattr(
        "imbalance_benchmark.hydra.guards.time.perf_counter", lambda: next(ticks)
    )

    guard.start_item()
    guard.finish_item()  # 10s item
    guard.start_item()
    guard.finish_item()  # 1s item

    assert guard._margin == 10.0
