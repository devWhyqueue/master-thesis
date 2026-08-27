from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from imbalance_benchmark.commands.tuning import wave


def test_lock_waits_for_a_holder_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".tune-wave.lock"
    path.mkdir()
    monkeypatch.setattr(wave, "LOCK_POLL_SECONDS", 0)
    monkeypatch.setattr(time, "sleep", lambda _: path.rmdir())

    with wave._submission_lock(path):
        assert path.is_dir()

    assert not path.exists()


def test_lock_gives_up_once_the_wait_budget_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".tune-wave.lock"
    path.mkdir()
    monkeypatch.setattr(wave, "LOCK_POLL_SECONDS", 0)
    monkeypatch.setattr(wave, "LOCK_WAIT_SECONDS", 0)

    with pytest.raises(RuntimeError, match="still busy"):
        with wave._submission_lock(path):
            pass

    assert path.is_dir()


def test_lock_left_by_a_killed_job_is_reclaimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".tune-wave.lock"
    path.mkdir()
    stale = time.time() - wave.LOCK_STALE_SECONDS - 1
    os.utime(path, (stale, stale))
    monkeypatch.setattr(wave, "LOCK_WAIT_SECONDS", 0)

    with wave._submission_lock(path):
        assert path.is_dir()

    assert not path.exists()
