# 04 — Crossed condition family

**Depends on:** 03 (which pool contrast is constructible), 01 (argmax guard). **Gates:** 07.
**Cluster cost:** full `freeze` re-run, and roughly +67 % on everything downstream.

## Context

Plan 03 establishes that the independent-support axis cannot be created by *narrowing* the evidence
pool — the pool is already the narrowest the contribution caps permit — and that it can be created by
*spreading* it. This plan builds the conditions that create it, so `beta_ind` is estimated from data
instead of fixed at zero by a predictor with no variance.

The first implementation of this plan (`15b0f35`) built a narrowed arm and is superseded. Most of its
structure survives: the condition plumbing, the per-axis deprivation fix, the guards, and the
enumeration sweep are all reusable. What inverts is which side of the contrast is new.

## The factorial

Two factors, crossed, per tail assignment:

| | pool = concentrated (existing) | pool = spread (new) |
| --- | --- | --- |
| **rho = 1** | `balanced` | **`balanced_spread`** |
| **rho = 10** | `moderate` | — |
| **rho = 100** | `severe` | **`severe_spread`** |

`moderate` stays in the concentrated arm as the intermediate nominal dose and gets no spread partner.

**`balanced_spread` alone is the minimum identifying cell.** `_cell_predictors`
([rq3_wiring.py:45-51](../experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/predictors/rq3_wiring.py#L45-L51))
uses `log(rho)` as the nominal column, so the `balanced` / `balanced_spread` contrast sits at
`log(rho) = 0` with `independent_shortage > 0` — orthogonal to every existing row. It also gives
`_label_unit` a unit where "independent" can legitimately win on a *positive* raw score.

**Add `severe_spread` too.** With only `balanced_spread` you get the independent main effect but
cannot test whether the two shortages are additive — and *"these shortages are not interchangeable"*
is precisely a claim about how they combine.

**Cost.** Going 3 → 5 controlled conditions is about **+67 %** of the controlled tuning and
confirmation budget; 3 → 4 is about +33 %. On the two wave-1 datasets that is 3,960 → 6,600 updates
for BRACS and 10,500 → 17,500 for TCGA-UT, both far below PANDA's 257,790 base.
**Documented fallback if that does not fit on a later dataset:** scope the spread arm to the `native`
assignment only. Tuning observations per spread shard drop from `|assignments|` to 1, and the matching
record gains 8 units instead of 18. Either is adequate for the pooled standardization. Unlike the
superseded narrowed arm, this scoping is now a *cost* decision, not a feasibility one — plan 03
measures every assignment of every dataset as constructible.

## Why the spread arm is constructible where the narrow arm was not

The two directions run into opposite bounds, and only one of them is occupied:

| | narrow arm (superseded) | spread arm |
| --- | --- | --- |
| Bound it pushes against | 10 % patient / 5 % slide contribution caps, i.e. ≥10 patients and ≥20 slides | eligible patient inventory |
| Distance to that bound | **zero** — `_expand_pool` already builds the minimal patient set | 1.0×–17.1× (plan 03, measured) |
| Effect on per-unit contribution | raises it, toward the caps | lowers it, away from the caps |
| Effect of a large `max(counts)` | makes narrowing harder | irrelevant; `max(counts)` exceeds eligible patients by orders of magnitude |

The last row is the one that matters for the datasets this benchmark cares about. `_pool_has_capacity`
requires the largest allocated count to exhaust every pool patient and slide, which caps the spread
pool at `max(counts)` patients — but `max(counts)` is in the thousands on every dataset (BRACS
allocates 2,400 per class at `balanced` alone) while eligible patients per class run from 30 to 514.
The constraint is slack everywhere.

## The estimand, and the one place this costs more than the narrow arm did

The concentrated pool has fewer independent units, so it is the **deprived** arm and the spread pool
is the **reference**:

> `D_ind = M(balanced_spread) - M(balanced)`

`balanced` is therefore the reference for the nominal axis and the *cell* for the independent axis.
That is coherent — it is the equal-allocation, minimal-pool condition, and it is genuinely deprived
relative to spread — but the plumbing hard-codes `balanced` as every condition's reference. The
superseded plan claimed "zero changes to gate and recovery plumbing"; that claim does not survive the
inversion and should not be repeated.

The change is contained: a per-condition reference map,

```python
CONDITION_REFERENCE = {
    "moderate": "balanced", "severe": "balanced",
    "balanced": "balanced_spread", "severe_spread": "balanced_spread",
}
```

consumed where `balanced_baseline` currently resolves its reference. One dict, one lookup, one test.

Two things offset it. `balanced` is already fitted for the full roster, so the independent contrast
**reuses existing runs on its deprived side** — only the spread conditions are new fits. And the
deficit reads directly as *"the deficit caused by independent-support loss alone at equal nominal
allocation"*, which is the estimand the benchmark currently cannot produce.

State the caveat in the protocol: the independent arm's deficit is a deliberate **between-pool**
contrast at fixed nominal allocation.

## The trap that would defeat the whole exercise

Unchanged from the superseded plan, and it still applies to the spread arm.

`_deprived_classes`
([rq3_features.py:42-51](../experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/predictors/rq3_features.py#L42-L51))
is defined by **nominal** count `<` balanced count, and every shortage averages over it. Between
`balanced` and `balanced_spread` no class is nominally deprived, so the set is empty and
`_independent_shortage` returns `0.0` — **defect A reproduced one level down, in the very cell built
to fix it.**

Give each shortage its own deprivation set: `_nominal_shortage` keeps the nominal set;
`_independent_shortage` uses `{c : n_patients_cell < n_patients_ref}`; `_diversity_shortage` uses the
union. Test this before anything is frozen.

## Which classes to spread

**All of them.** The concentrated pool is minimal for every class, so "spread every class to its
eligible inventory" is the natural opposite extreme and introduces no free parameter — there is no
`INDEPENDENT_SPREAD_RATIO` to justify, which is exactly where the superseded plan went wrong.

Classes whose eligible count already equals their concentrated count contribute zero to the arm's
shortage. Plan 03 measures one such class per split on BRACS and one on TCGA-UT; the per-axis
deprivation set above handles them correctly by excluding them from the mean rather than dragging it
toward zero.

Record `spread_classes` and the **achieved** per-class ratio in `condition_metadata`; the analysis
must read achieved, never requested. For the calibration endpoint, the spread arm's tail tier is the
`ceil(K/3)` classes with the largest achieved log-shortage — on the independent axis, the tail *is*
the set most deprived of independent evidence. `_tail_classes`
(`analysis/inference/context.py:186-211`) must prefer that recorded set over `assign_tiers`, which
falls back to an arbitrary tie-break when allocations are equal.

## Implementation

Delta from the landed narrowed-arm implementation (`15b0f35`), which is otherwise reused.

**Pool spreading** — [patch_pool.py](../experiments/2_benchmark_patch/code/imbalance_benchmark/manifest/sampling/patch_pool.py):

- **Revert** `max_independent_units`, its validation in `_validate_independent_units`, and the
  new-patient branch guard in `_expand_pool`. Nothing needs them now, and `_expand_pool`'s
  breadth-first order is correct as it stands.
- **Keep** `MIN_POOL_PATIENTS` / `MIN_POOL_SLIDES` and the cap-equivalence comment. They are still
  the floor; the spread arm simply never approaches it.
- No new parameter is required. `designate_patch_pool` already spreads when handed a larger
  `min_independent_units`: `_designate_floor` takes `pats[:min_independent_units]` from the same
  seeded shuffle, so the concentrated pool's patients are a **prefix** of the spread pool's. The
  spread pool is a strict patient **superset** of the concentrated one — the mirror of the
  superseded plan's nesting invariant, and verified in plan 03's canary on every measured cell.

**Pool designation** — `construction_helpers.py`: replace `designate_narrowed_patch_pools` and
`_designate_one_narrow_pool` with `designate_spread_patch_pools`, which passes
`min_independent_units = eligible patients of that class` and drops `INDEPENDENT_NARROW_RATIO`
entirely. Return achieved ratios exactly as before.

**Arm scoping** — `shared_total/narrowing.py` becomes `shared_total/spreading.py`:
`NARROWED_ASSIGNMENTS_BY_DATASET` becomes `SPREAD_ASSIGNMENTS_BY_DATASET` with every dataset at
`None` (every locked assignment), since plan 03 measures every assignment as constructible. Keep the
map as the documented cost knob for plan 08, not as a feasibility gate. `_narrowed_classes` is
deleted — the spread arm covers every class.

**Naming.** `balanced_spread` / `severe_spread`. As with the superseded names, anything outside the
`"balanced"` / `"natural"` pair flows through the assignment-scoped path unchanged, so the fourteen
call sites that branch on those literals generalize for free.

**Freeze wiring** — no signature change; a second `_build_conditions` call per assignment with the
spread names and spread pools. Register both names in `CONDITION_RHOS`. **Pass the spread frame as
`spec["pool"]`** — otherwise `pool_fraction_retained` is computed against the wrong denominator.

**Semantic changes, not just enumeration:**

- `reject_degenerate_conditions` must gate on `requested_rho != 1.0`, or it rejects `balanced_spread`
  outright — its `achieved_rho` is 1.0 and its counts equal balanced. Already landed; keep.
- `_get_constraints` (`manifest/freeze.py:220`): `if name == "balanced"` becomes
  `if CONDITION_RHOS.get(name, 1.0) == 1.0`. Already landed; keep.
- `reject_degenerate_narrowing` becomes `reject_degenerate_spreading`, rejecting any freeze whose
  spread arm mean achieved log-shortage falls below the floor plan 03 sets from measurement.
- New `reject_non_nested_pools`: the concentrated pool's patients must be a subset of the spread
  pool's, per class. Cheap, and it catches a seed or ordering regression that would silently turn the
  contrast into a patient *swap* rather than a patient *addition*.
- `reject_constant_signal_axes` — refusing any freeze in which one of the four shortage axes is
  constant across every comparison unit. This is the guard that would have caught the original
  defect; it is unchanged and is the single most valuable check in this plan.

**Pure enumeration** — already swept for the narrowed names in `15b0f35`; the rename carries through
`CONDITION_RHOS`, `context.CONDITIONS`/`CONTROLLED_CONDITIONS`, `signal_profile.py`,
`reporting/completeness.py`, `recovery.py`, `derive_deficit_thresholds.SEVERITIES`, `__main__.py`
`--condition` choices, `wave_round.py`, `report/sources.py`, `report/construction.py` (its
`_UNCHECKED` set and `_TOLERANCE` map), and `len(CONTROLLED_CONDITIONS) *` in `hydra/workflow.py`,
`hydra/dependent_jobs.py`, `hydra/resume.py`.

## Verification

Tests re-scoped rather than deleted, as before:

- `tests/commands/test_freeze.py:99` `test_freeze_uses_one_patch_pool_for_balanced_and_every_assignment`
  — `len(pool_hashes)` becomes 2.
- `tests/construction/test_construction.py:559,523` — pool-identity assertions.
- `tests/construction/test_construction.py:578` — `pool_fraction_retained == 1.0` passes for the
  spread arm **only if** `spec["pool"]` is the spread frame. Good canary for that exact mistake.
- `tests/commands/test_freeze.py:164,194` — a `requested_rho == 1.0` spread cell must not be rejected
  as degenerate.
- `tests/analysis/test_rq3.py:173` encodes the nominal-only deprivation set; changes with the split.

New tests:

- **Spread pool is a strict patient superset with unchanged nominal counts.**
- **Reference resolution**: `balanced` resolves to `balanced_spread`, `severe` to `balanced`.
- **Construction creates variation on the axis it claims to measure** — some class's `n_patients`
  differs across conditions, enforced at freeze time by `reject_constant_signal_axes`.

**End-to-end gate:** freeze BRACS first — cheapest, and plan 03's weakest measured dose, so it is the
arm most likely to fail a degeneracy check — and confirm `signal_profile.json` shows a **non-zero,
varying** `independent_shortage` before committing cluster time to TCGA-UT. That single check is the
point of the whole exercise; everything downstream is wasted if it comes back zero again.

## Done when

Both wave-1 datasets freeze with a spread arm, `reject_constant_signal_axes`,
`reject_degenerate_spreading` and `reject_non_nested_pools` all pass, and the matching record contains
at least one unit whose dominant shortage is independent support on a **positive raw score**.
