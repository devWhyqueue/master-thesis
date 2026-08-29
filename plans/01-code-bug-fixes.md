# 01 — Code bug fixes

**Depends on:** nothing. **Gates:** every other plan. **Cluster cost:** none.

## Context

Four defects produce wrong numbers in the published results. All are small, local edits. They come
first because re-running the pipeline against any of them wastes the whole cluster budget.

## B — a locked strength of zero trains at strength one

[training/__init__.py:101,104,107](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/__init__.py#L101-L107)
builds criteria with `float(param or 1.0)`. Python `0.0` is falsy, so it becomes `1.0`.

Tuning aliases strength-0 to CE's already-fitted metrics
([tuning_reduction.py:55-76](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/workflows/tuning/tuning_reduction.py#L55-L76)),
but confirmation has no such alias. So tuning selected *"focal degenerates to CE"* and confirmation
trained gamma = 1.0 at CE's learning rate: a configuration never validated.

13 selection units locked `0.0000` in `tuning_selections.tex` — 10 focal loss (all six TCGA-UT
conditions, both CAMELYON16 moderate, two balanced), plus PANDA balanced `balanced_sampling` and
BRACS balanced `ce_soft_f1`. This is the entire explanation for focal loss recovering nothing
anywhere (-0.161 to 0.111 across all 12 gate-passing units).

**Fix the `or` idiom. Do not alias CE's checkpoint in confirmation.**

Replace `float(param or 1.0)` with `float(param if param is not None else 1.0)` at all three sites —
the idiom already used correctly in
[signal_weights.py:43,47,52](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/signal_weights.py#L43-L52).

Aliasing in confirmation was the tempting fix and is the wrong one. The tuning alias is a compute
optimisation over *metrics*; confirmation must produce a real run record with its own
`test_prediction_sha256`, `selected_checkpoint_sha256` and cost
([confirmation_helpers.py:162-201](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/workflows/confirmation_helpers.py#L162-L201)).
Aliasing there would make the method's record a byte-duplicate of CE's — exactly the pathology of
defect C, reproduced deliberately. With `or` removed, `FocalLoss(gamma=0)` and weighted CE at
`counts^0` genuinely *are* CE, so the run trains the right objective and still yields its own artefact.

**Two companions:**

- Add a guard in `_init_criterion` raising when a locked config's `parameter` is `None` for a method
  whose `GRIDS` entry has a strength dimension. The bug was only reachable because a missing
  parameter silently became 1.0.
- **Do not "fix"** `loaders._base_sampler:57` (`if method == "balanced_sampling" and param:`). The
  falsy check is correct there — strength 0 gives `RandomSampler`, i.e. exactly CE. Changing it to
  `is not None` would substitute a with-replacement uniform sampler and break that method's CE
  anchor. Leave a comment so it is not later tidied away.

## B2 — the soft hybrids are anchored to the wrong baseline

`ce_soft_f1` and `ce_soft_mcc` are in `CE_ANCHORED_METHODS`
([search_windows.py:35-37](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/workflows/tuning/search_windows.py#L35-L37)),
so their strength-0 candidate is scored with CE's metrics. But both train under a **forced balanced
sampler** at strength 1.0, untuned
([loaders.py:28,59-60](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/loaders.py#L28-L60)).
Their strength-0 point is therefore balanced-sampling CE, not CE, and the alias substitutes the wrong
metrics. BRACS balanced `ce_soft_f1` locked `0.0000` on that basis.

Drop both from `CE_ANCHORED_METHODS`.

## C — post-hoc logit adjustment discards its tau on the calibration axis

`target_prior_logits` ([calibration.py:190-197](../experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/calibration.py#L190-L197))
routes `post_hoc_logit_adjustment` through CE's branch. Post-hoc LA reuses CE's checkpoint, so its
probabilities come out byte-identical to CE's and its tail-NLL recovery is exactly
`0.000 [0.000, 0.000]` in all six calibration-gate units — a degenerate CI because the paired
difference is identically zero in every bootstrap draw. All six selected tau = 0.5, so the bug is
active in every one. `balanced_decision_logits` fifteen lines below *does* apply tau, so the
discrimination axis is unaffected.

**Fix**, expressed *through* the sibling so the two cannot drift apart again:

```python
if method == "ce":
    return logits - log_train + log_target
if method == "post_hoc_logit_adjustment":
    return balanced_decision_logits(logits, method, tau, pi_train) + log_target
return logits + (tau - 1.0) * log_train + log_target
```

`confirmation_helpers.py:93-94` already passes the same `tau` to both functions, so no call site
changes.

## A (labelling half) — the unguarded dominant-shortage argmax

`_label_unit` ([matching.py:109-126](../experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/predictors/signals/matching.py#L109-L126))
sorts the four standardized scores descending with no sign guard and no zero-variance exclusion.
`_standardize` ([matching.py:91-93](../experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/predictors/signals/matching.py#L91-L93))
maps a constant column to exactly `0.0`, so the structurally-zero independent-support column *wins*
whenever the other three scores are negative — which is how BRACS and TCGA-UT difficulty-reversed
moderate came to be labelled "Independent support dominant".

**Two independent fixes, both wanted:**

1. `_standardize` returns `NaN` for a degenerate column rather than zeros, so a never-varying axis is
   distinguishable from "average". Exclude NaN columns from the ranking and record them per unit as
   `degenerate_axes`.
2. `_label_unit` receives the **raw** scores alongside the standardized ones and filters the ranking
   to axes with raw score > 0 before both the argmax and the 0.25-SD ambiguity check. A shortage that
   was never created cannot be dominant. No positive axis means `dominant = None`.

The design half — actually *creating* the shortage — is plans 03 and 04.

## Verification

- **B:** `tests/modeling/test_training.py:294` parametrizes methods but never with `param=0.0`. Add a
  test asserting `_init_criterion("focal", 0.0, ...).gamma == 0.0` and `weighted_ce` at 0.0 yields
  all-ones weights. The bug is a falsy zero, so the test **must** use `0.0`, not `None`. Add a
  table-driven companion over `GRIDS` keys so a newly rostered method cannot regress.
- **C:** `tests/analysis/test_calibration.py:60` `test_target_prior_correction_posthoc_formula`
  **passes today while the bug is live**, because it tests at `tau = 1.0` where the buggy and correct
  formulas coincide. Re-parameterize at `tau = 0.5`, and add a variation invariant (outputs differ
  across `tau in (0.25, 0.5, 2.0)`) so a future refactor cannot drop tau silently.
- **A:** `tests/analysis/test_matching.py:108` `test_no_deprived_class_zero_scores_standardize_without_crashing`
  asserts `dominant in {None, "nominal", "independent", ...}` — it **explicitly tolerates the bug**.
  Tighten it: an axis that never varies across the population is never `dominant`. Three lines of
  fixture reproduce the published mislabelling.
- **B2:** `tests/modeling/test_tuning_reduction.py:127,157` re-scoped once the hybrids leave
  `CE_ANCHORED_METHODS`.

Run `clean-code` with the appropriate `--scope`, then only the affected tests —
`tests/modeling/test_training.py`, `tests/modeling/test_tuning_reduction.py`,
`tests/analysis/test_calibration.py`, `tests/analysis/test_matching.py`.

## Done when

All four fixes land, each new or amended test fails on the pre-fix code and passes after, and
`clean-code` is green.
