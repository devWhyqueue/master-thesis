from __future__ import annotations

import pytest

from imbalance_benchmark.hydra.queue import check_queue_cap


def test_check_queue_cap_passes_when_under_cap():
    check_queue_cap(cap=25, count=lambda: 24)


def test_check_queue_cap_raises_at_cap():
    with pytest.raises(RuntimeError, match="Queue at 25"):
        check_queue_cap(cap=25, count=lambda: 25)


def test_check_queue_cap_raises_over_cap():
    with pytest.raises(RuntimeError, match="cap 25"):
        check_queue_cap(cap=25, count=lambda: 40)


def test_check_queue_cap_uses_the_default_cap_of_25():
    with pytest.raises(RuntimeError, match="cap 25"):
        check_queue_cap(count=lambda: 25)
