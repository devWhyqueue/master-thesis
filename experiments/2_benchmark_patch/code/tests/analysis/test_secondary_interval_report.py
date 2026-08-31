from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imbalance_benchmark.analysis.reporting.secondary_intervals import interval_cache


def test_chunk_keys_splits_round_robin_and_drops_empty_workers() -> None:
    keys = [("a", "natural", "ce"), ("a", "natural", "oko"), ("a", "balanced", "ce")]

    chunks = interval_cache._chunk_keys(keys, 5)

    assert chunks == [[keys[0]], [keys[1]], [keys[2]]]


def test_key_cache_round_trips_through_atomic_write(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "a__natural__ce.npz"
    values = {"balanced_accuracy": np.array([0.5, 0.4, 0.6])}

    interval_cache._write_key_cache_atomic(path, values)
    loaded = interval_cache._read_key_cache(path)

    assert path.exists()
    assert not list(path.parent.glob(".*.tmp"))
    assert np.array_equal(loaded["balanced_accuracy"], values["balanced_accuracy"])


def test_cache_dir_is_namespaced_by_replicates_and_seed(tmp_path: Path) -> None:
    base_paths = {"data": tmp_path}

    assert interval_cache._cache_dir(base_paths, 1000, 7) != interval_cache._cache_dir(
        base_paths, 2000, 7
    )
    assert interval_cache._cache_dir(base_paths, 1000, 7) != interval_cache._cache_dir(
        base_paths, 1000, 8
    )


def test_compute_key_caches_skips_a_key_already_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = ("a", "natural", "ce")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(interval_cache, "BootstrapContext", lambda *_a, **_k: object())
    monkeypatch.setattr(interval_cache, "split_paths", lambda base, index: base)
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        interval_cache,
        "_split_distributions",
        lambda *args: calls.append(args[-3:]) or [{"endpoint": np.array([0.1])}],
    )

    interval_cache._compute_key_caches([key], {}, False, False, 10, 0, cache_dir)
    assert calls == [key]
    assert interval_cache._cache_path(cache_dir, key).exists()

    interval_cache._compute_key_caches([key], {}, False, False, 10, 0, cache_dir)

    assert calls == [key]  # second call found the cache file and skipped recompute


def test_distributions_by_key_spawns_no_workers_when_fully_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_paths = {"data": tmp_path}
    keys = [("a", "natural", "ce")]
    monkeypatch.setattr(interval_cache, "_complete_result_keys", lambda *_a, **_k: keys)
    cache_dir = interval_cache._cache_dir(base_paths, 10, 0)
    interval_cache._write_key_cache_atomic(
        interval_cache._cache_path(cache_dir, keys[0]), {"endpoint": np.array([0.1])}
    )

    class PoisonedContext:
        def Process(self, **_kwargs: object) -> None:
            raise AssertionError("no worker should spawn when every key is cached")

    monkeypatch.setattr(
        interval_cache.multiprocessing, "get_context", lambda *_a: PoisonedContext()
    )

    distributions = interval_cache.distributions_by_key(base_paths, False, False, 10, 0)

    assert set(distributions) == set(keys)
    assert np.array_equal(distributions[keys[0]]["endpoint"], np.array([0.1]))


def test_distributions_by_key_worker_failure_reaches_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_paths = {"data": tmp_path}
    keys = [("a", "natural", "ce")]
    monkeypatch.setattr(interval_cache, "_complete_result_keys", lambda *_a, **_k: keys)

    class FailedProcess:
        exitcode = 1

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class FailedContext:
        def Process(self, **_kwargs: object) -> FailedProcess:
            return FailedProcess()

    monkeypatch.setattr(
        interval_cache.multiprocessing, "get_context", lambda *_a: FailedContext()
    )

    with pytest.raises(RuntimeError, match="workers failed"):
        interval_cache.distributions_by_key(base_paths, False, False, 10, 0)
