"""The matched (semantic_scale_ce) vs. unmatched (weighted_ce) contrast at 'narrow'."""

from __future__ import annotations

from typing import Any

from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.permutation import (
    paired_block_permutation_ba,
    paired_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.query import load_seed_predictions

from diversity.analyze.combine import fisher_combine
from diversity.analyze.common import (
    MATCHED_METHOD,
    N_PERMUTATIONS,
    N_REPLICATES,
    UNMATCHED_METHOD,
    fixed_tail_classes,
    iter_splits,
)

__all__ = ["matched_vs_unmatched"]


def _split_p(
    exp3_paths: dict[str, Any], freeze: dict[str, Any], allocation: str, endpoint: str
) -> float | None:
    class_names = list(freeze["class_names"])
    try:
        matched = load_seed_predictions(
            exp3_paths, allocation, MATCHED_METHOD, assignment="narrow"
        )
        unmatched = load_seed_predictions(
            exp3_paths, allocation, UNMATCHED_METHOD, assignment="narrow"
        )
    except RuntimeError:
        return None
    if matched is None or unmatched is None:
        return None
    ctx = BootstrapContext(exp3_paths, False, N_REPLICATES, seed=0)
    if endpoint == "ba":
        return float(
            paired_block_permutation_ba(
                unmatched["labels"],
                matched["preds"],
                unmatched["preds"],
                ctx.case_ids,
                len(class_names),
                N_PERMUTATIONS,
            )
        )
    tail_classes = fixed_tail_classes(freeze, class_names)
    if not tail_classes:
        return None
    return float(
        paired_block_permutation_tail_nll(
            unmatched["labels"],
            matched["probs"],
            unmatched["probs"],
            ctx.case_ids,
            tail_classes,
            N_PERMUTATIONS,
        )
    )


def _cell_contrast(
    config: dict[str, Any], allocation: str, endpoint: str
) -> dict[str, Any] | None:
    per_split_p = [
        p
        for _, exp3_paths, freeze in iter_splits(config)
        if (p := _split_p(exp3_paths, freeze, allocation, endpoint)) is not None
    ]
    if not per_split_p:
        return None
    return {
        "allocation": allocation,
        "endpoint": endpoint,
        "p_value": fisher_combine(per_split_p),
        "status": "tested",
    }


def matched_vs_unmatched(
    config: dict[str, Any], damage: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run only in cells where the corresponding damage gate opened (report Sec. "Analysis")."""
    reports = []
    for entry in damage:
        if entry.get("gate_passed"):
            contrast = _cell_contrast(config, entry["allocation"], entry["endpoint"])
            if contrast is not None:
                reports.append(contrast)
    return reports
