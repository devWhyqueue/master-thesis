"""Stage 4 (analyze, CPU): damage, interaction, and recovery, with Holm-adjusted gates.

Not run in this task (Stage 3 has not been submitted) -- implemented and
structured per plan Stage 4, reusing exp-2's bootstrap/permutation/gate
machinery verbatim wherever the plan names a function to reuse.

Cross-split combination note (judgment call, see the accompanying report):
the plan says results "combine[] with equal weight" across the three
patient splits, mirroring exp-2's own separate ``analyze`` (per-split) /
``analyze-combine`` (equal-weight combine) split, but does not pin down the
exact combination formula for exp-3 and this subpackage does not reuse
exp-2's own ``analyze-combine`` internals bit-for-bit. Each split's own
effect/CI/p-value is computed with exp-2's real per-split machinery
(``damage.py``), then combined across splits with a documented, simple
equal-weight rule (mean point estimate, pooled recentred bootstrap
replicates for the CI, Fisher's method for the p-value; ``combine.py``).

Facade over ``common`` (shared constants/plumbing), ``combine`` (cross-split
combination), ``thresholds`` (deficit-gate sigma_seed), ``damage``
(RQ-D1), ``interaction`` (RQ-D2), ``recovery`` (RQ-D3), ``matched_contrast``,
and ``holm_family`` -- kept small per this repo's file-length rule.
"""

from __future__ import annotations

import logging
from typing import Any

from imbalance_benchmark.analysis.inference.confirmatory.holm import holm_adjust_pvalues
from imbalance_benchmark.common import ensure_dirs, write_json

from diversity.analyze.damage import damage_report
from diversity.analyze.holm_family import confirmatory_pvalues
from diversity.analyze.interaction import interaction_report
from diversity.analyze.matched_contrast import matched_vs_unmatched
from diversity.analyze.recovery import recovery_report

__all__ = ["run_analyze"]

logger = logging.getLogger(__name__)


def _holm_table(family: list[tuple[str, float]]) -> list[dict[str, Any]]:
    adjusted = holm_adjust_pvalues([p for _, p in family]) if family else []
    return [
        {"label": label, "p_value": p, "adjusted_p_value": adj}
        for (label, p), adj in zip(family, adjusted)
    ]


def run_analyze(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full damage / interaction / recovery analysis for one dataset."""
    dataset = config["dataset"]["name"]
    damage = damage_report(config, dataset)
    interaction = interaction_report(damage)
    matched = matched_vs_unmatched(config, damage)
    recovery = recovery_report(config, dataset, damage)
    holm = _holm_table(confirmatory_pvalues(damage, interaction, matched))
    summary = {
        "dataset": dataset,
        "damage": [
            {k: v for k, v in d.items() if k != "contribution_vectors"} for d in damage
        ],
        "interaction": interaction,
        "matched_vs_unmatched": matched,
        "recovery": recovery,
        "confirmatory_holm": holm,
    }
    out_path = ensure_dirs(config)["tables"] / "diversity_analysis.json"
    write_json(out_path, summary)
    logger.info("analyze: wrote %s", out_path)
    return summary
