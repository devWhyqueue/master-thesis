"""Checks for the canonical, split-independent class order and its permutations."""

from __future__ import annotations

import numpy as np

import breadth.analyze.canonical as canonical


def _fake_freeze(names_by_split: dict[int, list[str]]):
    def _load(exp2_paths: dict[str, int]) -> dict[str, list[str]]:
        return {"class_names": names_by_split[exp2_paths["split"]]}

    return _load


def _patch(monkeypatch, names_by_split: dict[int, list[str]]) -> None:
    monkeypatch.setattr(canonical, "exp2_split_paths", lambda config, s: {"split": s})
    monkeypatch.setattr(canonical, "load_freeze_meta", _fake_freeze(names_by_split))


def test_canonical_class_names_is_sorted(monkeypatch):
    _patch(monkeypatch, {0: ["b", "a", "c"]})
    assert canonical.canonical_class_names({}) == ["a", "b", "c"]


def test_canonical_permutation_reorders_local_rows(monkeypatch):
    """A split whose own freeze order differs from the canonical one still yields
    the right class's row when its recall array is gathered by the permutation."""
    _patch(monkeypatch, {0: ["a", "b", "c"], 1: ["b", "c", "a"]})
    canonical_names = canonical.canonical_class_names({})

    perm = canonical.canonical_permutation({}, 1, canonical_names)
    local_rows = np.array([10, 20, 30])  # split 1's own order: b=10, c=20, a=30
    assert list(local_rows[perm]) == [30, 10, 20]  # canonical order: a, b, c


def test_canonical_permutation_identity_when_orders_match(monkeypatch):
    _patch(monkeypatch, {0: ["a", "b", "c"], 2: ["a", "b", "c"]})
    canonical_names = canonical.canonical_class_names({})
    perm = canonical.canonical_permutation({}, 2, canonical_names)
    assert list(perm) == [0, 1, 2]
