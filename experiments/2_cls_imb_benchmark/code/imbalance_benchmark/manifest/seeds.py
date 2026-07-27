from __future__ import annotations

import hashlib

__all__ = ["SEED_ROLES", "derive_seed"]

SEED_ROLES = (
    "patient_split",
    "patient_split_0",
    "patient_split_1",
    "patient_split_2",
    "assignment",
    "pilot_construction_0",
    "pilot_construction_1",
    "pilot_construction_2",
    "definitive_construction",
    "initialization",
    "resampling",
    "tuning_initialization_0",
    "tuning_initialization_1",
    "confirmation_initialization_0",
    "confirmation_initialization_1",
    "confirmation_initialization_2",
    "confirmation_initialization_3",
    "confirmation_initialization_4",
)


def derive_seed(base_seed: int, role: str) -> int:
    """Derive a role-specific seed, keeping the five seed families disjoint.

    Each role hashes to an independent stream from the same base seed, so
    patient-split, assignment, construction, initialization, and resampling
    seeds never collide even though they share one user-provided base seed.
    """
    if role not in SEED_ROLES:
        raise ValueError(f"Unknown seed role {role!r}; expected one of {SEED_ROLES}")
    digest = hashlib.sha256(f"{base_seed}:{role}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
