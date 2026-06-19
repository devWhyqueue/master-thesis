import os
from typing import Literal, overload


@overload
def gather_results_across_seeds(
    path: str, return_n_seeds: Literal[False] = False
) -> list[str]:
    """Collect validation result paths below seed folders."""
    ...


@overload
def gather_results_across_seeds(
    path: str, return_n_seeds: Literal[True]
) -> tuple[list[str], int]:
    """Collect validation result paths and the seed count."""
    ...


def gather_results_across_seeds(
    path: str, return_n_seeds: bool = False
) -> list[str] | tuple[list[str], int]:
    """Collect validation result paths below seed folders."""
    result_paths = [_result_path(path, seed_folder) for seed_folder in os.listdir(path)]
    if return_n_seeds:
        return result_paths, len(result_paths)
    return result_paths


def _result_path(path: str, seed_folder: str) -> str:
    seed_path = os.path.join(path, seed_folder)
    children = os.listdir(seed_path)
    if len(children) == 1:
        return os.path.join(seed_path, children[0], "validation_results.json")
    return os.path.join(seed_path, "validation_results.json")
