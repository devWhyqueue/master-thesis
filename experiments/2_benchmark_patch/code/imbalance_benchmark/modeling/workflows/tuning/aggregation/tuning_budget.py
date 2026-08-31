from __future__ import annotations

from imbalance_benchmark.modeling.context import Regime

__all__ = ["TUNING_BUDGET_FRACTION", "tuning_example_budget"]

# Item 4 (after-a-first-run-linear-wave): tuning fits only rank candidates, so
# they train to this fraction of confirmation's full E = 30T budget instead of
# the full exposure. A BRACS dense-trace replay (12 base methods x 2
# conditions, full candidate grids) found the selected candidate changed for
# only 4.2% of (condition, method) groups at this fraction, mean BA loss
# ~0.0001 -- comfortably inside the plan's >=95%-unchanged bar. Confirmation
# reads ``exposure_budgets`` directly (commands/confirm), never through this
# helper, so it keeps the full frozen budget untouched.
TUNING_BUDGET_FRACTION = 0.40


def tuning_example_budget(regime: Regime, condition: str) -> int | None:
    """Truncated example budget for one tuning scope, or None if unset."""
    full = regime.exposure_budgets.get(
        "natural" if condition == "natural" else "controlled"
    )
    return round(full * TUNING_BUDGET_FRACTION) if full is not None else None
