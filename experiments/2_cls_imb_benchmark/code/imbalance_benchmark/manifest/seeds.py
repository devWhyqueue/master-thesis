from __future__ import annotations

import hashlib

__all__ = ["SEED_ROLES", "derive_seed"]

SEED_ROLES = (
    "patient_split",
    "assignment",
    "pilot_construction_0",
    "pilot_construction_1",
    "pilot_construction_2",
    "definitive_construction",
    "initialization",
    "resampling",
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
