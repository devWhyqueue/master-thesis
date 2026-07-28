from __future__ import annotations


def _bundle_indices(
    bundle_index: int, size: int, observation_count: int, by_observation: bool
) -> list[int]:
    if not by_observation:
        first = bundle_index * size
        return list(range(first, first + size))
    candidate_group, observation_index = divmod(bundle_index, observation_count)
    first_candidate = candidate_group * size
    return [
        candidate_index * observation_count + observation_index
        for candidate_index in range(first_candidate, first_candidate + size)
    ]
