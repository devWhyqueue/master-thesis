"""Controlled class-imbalance construction targets at patch and slide level.

Applies truncated exponential decay with a floor (Cui et al., 2019), step
imbalance (Buda et al., 2018), and ratio downsampling (binary case) to each
dataset's class counts, independently at patch and slide level, producing the
native-vs-target supports the report figures and table consume. Datasets are
returned in the Table-1 order (CAMELYON16, TCGA-UT, BRACS, PANDA).
"""

from __future__ import annotations

# Slide floor: the 70/15/15 split leaves >= 0.15*20 = 3 test slides per class.
# Patch floor: >= 0.15*200 = 30 test patches per class for stable per-class recall.
SLIDE_FLOOR = 20
PATCH_FLOOR = 200
# Binary patch targets are pushed to the long-tail stress-test regime (rho ~ 100).
PATCH_RATIO = 100

# PANDA's slide construction uses the FULL 10,616-slide cohort (train.csv), not the
# 2000-slide feature-extraction subset; its patch counts are only the subset, but
# rho and H_norm are proportion-based so the ratios are unaffected.
PANDA_FULL_SLIDES = {
    "ISUP0": 2892,
    "ISUP1": 2666,
    "ISUP2": 1343,
    "ISUP3": 1242,
    "ISUP4": 1249,
    "ISUP5": 1224,
}
ISUP_ORDER = ["ISUP0", "ISUP1", "ISUP2", "ISUP3", "ISUP4", "ISUP5"]
# BRACS step-imbalance head group (benign/usual); the atypical/malignant rest is the tail.
BRACS_HEAD = {"N", "PB", "UDH", "IC"}

Panel = dict[str, object]


def _exp_decay(ordered: list[int], n_floor: int) -> list[int]:
    """Cui et al. truncated exponential decay: n_i = max(floor, N_max * mu**i)."""
    n_max, k = ordered[0], len(ordered)
    mu = (n_floor / n_max) ** (1 / (k - 1))
    return [min(n, max(n_floor, round(n_max * mu**i))) for i, n in enumerate(ordered)]


def exp_decay_panel(
    counts: dict[str, int],
    n_floor: int,
    order: list[str] | None = None,
    method: str = "Exp.\\ decay",
) -> Panel:
    """Exponential-decay panel, ranked by support unless a fixed ``order`` is given."""
    labels = order or sorted(counts, key=lambda c: -counts[c])
    native = [counts[c] for c in labels]
    return {
        "method": method,
        "labels": labels,
        "native": native,
        "target": _exp_decay(native, n_floor),
        "floor": n_floor,
    }


def step_panel(counts: dict[str, int], head: set[str], n_floor: int) -> Panel:
    """Buda et al. step imbalance: head keeps native support, tail is floored."""
    labels = sorted(counts, key=lambda c: -counts[c])
    native = [counts[c] for c in labels]
    target = [n if c in head else min(n, n_floor) for c, n in zip(labels, native)]
    return {
        "method": "Step",
        "labels": labels,
        "native": native,
        "target": target,
        "floor": n_floor,
    }


def ratio_panel(
    counts: dict[str, int],
    major: str,
    minor: str,
    ratio: int,
    n_floor: int,
) -> Panel:
    """Downsample the minority class to a fixed major:minor ratio, floored."""
    n_major, n_minor = counts[major], counts[minor]
    target_minor = min(n_minor, max(n_floor, round(n_major / ratio)))
    return {
        "method": f"Ratio ({ratio}:1)",
        "labels": [major, minor],
        "native": [n_major, n_minor],
        "target": [n_major, target_minor],
        "floor": n_floor,
    }


def _camelyon(tile: dict, slide: dict) -> dict[str, Panel]:
    return {
        "patch": ratio_panel(tile, "normal", "tumor", PATCH_RATIO, PATCH_FLOOR),
        "slide": ratio_panel(slide, "normal", "tumor", 10, SLIDE_FLOOR),
    }


def _tcga(tile: dict, slide: dict) -> dict[str, Panel]:
    return {
        "patch": exp_decay_panel(tile, PATCH_FLOOR),
        "slide": exp_decay_panel(slide, SLIDE_FLOOR),
    }


def _bracs(tile: dict, slide: dict) -> dict[str, Panel]:
    return {
        "patch": step_panel(tile, BRACS_HEAD, PATCH_FLOOR),
        "slide": step_panel(slide, BRACS_HEAD, SLIDE_FLOOR),
    }


def _panda(tile: dict) -> dict[str, Panel]:
    return {
        "patch": ratio_panel(tile, "benign", "cancer", PATCH_RATIO, PATCH_FLOOR),
        "slide": exp_decay_panel(
            PANDA_FULL_SLIDES, SLIDE_FLOOR, ISUP_ORDER, "Ordinal decay"
        ),
    }


def build_targets(rows: list[dict]) -> dict[str, dict[str, Panel]]:
    """Return {dataset: {"patch": panel, "slide": panel}} in Table-1 order."""
    tile = {r["dataset"]: r["tile"]["counts"] for r in rows}
    slide = {r["dataset"]: r["slide"]["counts"] for r in rows}
    return {
        "CAMELYON16": _camelyon(tile["CAMELYON16"], slide["CAMELYON16"]),
        "TCGA-UT": _tcga(tile["TCGA-UT"], slide["TCGA-UT"]),
        "BRACS": _bracs(tile["BRACS"], slide["BRACS"]),
        "PANDA": _panda(tile["PANDA"]),
    }


def _self_check() -> None:
    """Sanity-check the constructions: floors respected, targets never exceed native."""
    decay = _exp_decay([792, 400, 100, 28], 20)
    assert decay[0] == 792 and all(
        20 <= t <= n for t, n in zip(decay, [792, 400, 100, 28])
    )
    ratio = ratio_panel({"a": 1000, "b": 300}, "a", "b", 100, 20)
    assert ratio["target"] == [1000, 20], (
        "minority driven to floor when ratio too steep"
    )
    step = step_panel({"a": 200, "b": 90, "c": 40}, {"a"}, 30)
    assert step["target"] == [200, 30, 30], "head kept, tail floored"


if __name__ == "__main__":
    _self_check()
