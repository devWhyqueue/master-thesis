# 06 — Method implementation fixes

**Depends on:** 01. **Gates:** 07. **Cluster cost:** none directly.
Independent of 04 and 05 — can proceed in parallel.

## Context

Three roster methods have implementation details that make their published numbers hard to interpret.
None is as clear-cut as the plan-01 bugs, but each one sits underneath a result the report currently
states as a finding.

## 1. Semantic-scale weighting has a 1000x degenerate fallback

`semantic_volumes` skips any class whose computed volume is exactly `0.0`
([semantic_scale.py:153](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/semantic_scale.py#L153)),
so that class keeps a stale or absent entry. `_ssb_weights` then falls back to `EPS_S = 1e-3`
([semantic_scale.py:171-175](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/semantic_scale.py#L171-L175)),
giving `1e-3 ** (-tau)` — up to **1000×** before mean-one normalization. `_update_volumes` counts a
class with fewer than two filled rows as an invalid draw, which is exactly the situation a severe
condition's tail class is in.

`ssb_invalid_draws` is recorded in `method_diagnostics` and **never surfaced anywhere**. Diversity
weighting's erratic pattern across the benchmark — 0.758 on PANDA moderate, -0.004 on PANDA severe,
0.655 / 0.509 on CAMELYON16, roughly zero on BRACS and TCGA-UT — is exactly what an intermittent
1000× weight spike would produce.

**The diversity-weighting null result cannot be believed until that counter is read.** Surface it per
condition in the report appendix. If it is non-zero anywhere a diversity claim is made, fix the
fallback (a degenerate volume should mean *no reweighting*, not maximal reweighting) and re-run.

Semantic-scale is also the only method with a warmup — unit weights for passes 1-5 of 30
([semantic_scale.py:190](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/semantic_scale.py#L190)).
Document it as a method-specific deviation or remove it.

## 2. CFAL's spec and code disagree

- Three loss constants are hard-coded and never swept: `gamma=2.0, beta=0.999, margin=0.1`
  ([losses.py:123-131](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/losses.py#L123-L131)).
  Only the affinity bandwidth `sigma` is tuned.
- The docstring says the prototype-diversity term uses `lambda = 0.1`, but the code adds it at
  coefficient **1.0**: `return loss_cfal + _prototype_diversity(model)`
  ([losses.py:144](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/losses.py#L144)).
- CFAL alone skips mean-one weight normalization, flagged in its own docstring at `losses.py:132`.

CFAL is the worst performer in the benchmark on PANDA (-0.683 and -0.463, tight CIs). That number is
not reportable until it is settled whether it reflects the method or a 10× diversity coefficient.
Resolve the mismatch — code to match the spec, or spec to match the code — and say which.

## 3. Focal loss is the alpha-free variant

`FocalLoss.__init__` accepts `alpha` ([losses.py:31](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/losses.py#L31)),
but the only construction site is `FocalLoss(gamma=float(param or 1.0))`
([training/__init__.py:104](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/__init__.py#L104)) —
`alpha` stays `None`, so the class-weight branch at `losses.py:44-45` never runs. Focal also gets a
plain `RandomSampler` and no logit correction.

As implemented it is therefore a **pure difficulty reweighter with no class awareness**. Its near-zero
recovery is then entirely consistent with pilot-difficulty CE also recovering nothing — a coherent
finding, not an anomaly. But the protocol lists focal under "Difficulty" as an imbalance method
without saying which variant, and the original formulation's recommended configuration for imbalance
is alpha-balanced.

This is a **documentation fix, not a code fix**, unless the alpha-balanced variant is wanted as a
separate roster entry. State the variant in the protocol either way.

## Ordering note, under wave 1 (BRACS first, then TCGA-UT)

Item 1 is the only one that can invalidate a published claim, so do it first — and BRACS is the right
place to read the counter. Diversity weighting recovered essentially nothing on BRACS across all six
units (-0.022 to 0.370 on discrimination), which is precisely the pattern an intermittent 1000× weight
spike would produce, and BRACS is the only dataset where the calibration axis exists to see the damage
on. If `ssb_invalid_draws` is non-zero on BRACS, the diversity null is an artefact and plan 07's
five-family conclusion cannot be written until the fallback is fixed and BRACS retuned.

Item 2 (CFAL) loses urgency: its only alarming number (-0.683, -0.463) was on PANDA, which plan 08
may never run and which is the last thing scheduled if it does. Settle the spec/code mismatch as
**documentation** now, and defer any behavioural change until a dataset that exercises it is actually
scheduled. On BRACS, CFAL recovered 0.268 to 0.513 — unremarkable, and not evidence of a defect.

Item 3 is documentation only and can be done any time.

## Verification

- Assert `ssb_invalid_draws` is emitted into the run record and reaches the report appendix.
- Add a test that a class with a degenerate (zero) semantic volume receives a weight of 1 after
  mean-one, not the `EPS_S` maximum — this is the assertion whose absence let the fallback ship.
- `tests/modeling/test_semantic_scale.py:116` `test_ssb_uses_unit_weights_for_exactly_five_passes`
  already pins the warmup; keep it and reference it from the protocol.
- Assert CFAL's prototype-diversity coefficient matches the documented value, whichever is chosen.

## Done when

`ssb_invalid_draws` is visible per condition and either zero or accounted for on BRACS, CFAL's
documented and implemented constants agree on paper, and the protocol names the focal variant it
tested.
