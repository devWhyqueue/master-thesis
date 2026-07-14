from __future__ import annotations


def method_floor(patient_equals_slide: bool) -> dict[str, int]:
    """Return the fixed method-floor independent-unit minimums per class."""
    return {"slides": 20} if patient_equals_slide else {"patients": 10, "slides": 20}


def meets_method_floor(support: dict[str, int], patient_equals_slide: bool) -> bool:
    """Require every independent-unit floor applicable to the regime."""
    return all(
        support.get(unit, 0) >= minimum
        for unit, minimum in method_floor(patient_equals_slide).items()
    )
