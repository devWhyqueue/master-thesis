from __future__ import annotations

from pathlib import Path


def tuning_result_dir(
    paths: dict[str, Path], benchmark: str, method: str, tuning_id: str, seed: int
) -> Path:
    """Return the isolated result directory for one tuning run."""
    return (
        paths["root"]
        / "outputs"
        / "tuning"
        / benchmark
        / method
        / tuning_id
        / f"seed={seed}"
    )
