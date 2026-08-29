# 03 — Independent-support feasibility

**Depends on:** nothing (can run alongside 01/02). **Gates:** 04.
**Cluster cost:** read-only measurement jobs. **Not a pilot re-run.**

## Context

The benchmark's four-axis taxonomy includes an independent-support axis — how many patients or slides
back a class. It is identically zero in all 18 comparison units, so one of the four axes was held
constant rather than varied, and the method that reads it was tested on conditions where the quantity
it reads does not move.

The mechanism is three design choices compounding:

1. One per-class pool is designated once
   ([freezing.py:179-185](../experiments/2_benchmark_patch/code/imbalance_benchmark/manifest/freezing.py#L179-L185),
   [construction_helpers.py:157-180](../experiments/2_benchmark_patch/code/imbalance_benchmark/manifest/construction_helpers.py#L157-L180))
   and handed to the balanced condition and every imbalanced condition alike.
2. `_pool_has_capacity`
   ([patch_pool.py:52-92](../experiments/2_benchmark_patch/code/imbalance_benchmark/manifest/sampling/patch_pool.py#L52-L92))
   *requires* the largest allocation to exhaust every pool patient and slide.
3. `_loop_patches` draws round-robin, one patch per patient per pass, so even a severe allocation
   still touches every pool patient.

Hence `_independent_shortage` = `mean(log(n_patients_balanced / n_patients_imbalanced))` = `log(1)` = 0.

## Why this is a separate plan

**The approved manipulation may not be achievable, and that is a measurement.** Locking a grid by
analogy with the nominal `{10, 100}`, without checking against the real construction, would repeat
the original mistake in a new place. It did: see the superseded result below.

### The floor is not negotiable — this is settled, not open

`MIN_POOL_PATIENTS = 10` / `MIN_POOL_SLIDES = 20`
([patch_pool.py:22-23](../experiments/2_benchmark_patch/code/imbalance_benchmark/manifest/sampling/patch_pool.py#L22-L23))
look like tunable safety margins. They are not. `_contribution_cap` caps any single patient at 10 %
and any slide at 5 % of a class's patches, so *any* allocation of n patches already requires at least
10 patients and 20 slides. The floors and the caps are the same constraint written twice. Relaxing
`_pool_is_ready` would only move the failure into `_select_from_hierarchy`'s `ValueError`.

## Superseded result: narrowing is infeasible by construction, on every dataset

The first version of this plan proposed creating the shortage by **narrowing** the pool — holding the
patch count fixed and removing patients — and set `INDEPENDENT_NARROW_RATIO = 0.55` from a per-class,
per-assignment proxy. Plan 04 was implemented against that decision (`15b0f35`) and the real crossed
construction refuted it on BRACS, the dataset the proxy had ranked as the *strongest* carrier:

- **12 of 21 `(class, split)` cells admit no narrowing at all** — the concentrated pool already sits
  at the independent floor.
- The remaining 9 shave only to ratios of **0.857–0.968**, against the plan's 0.55 target. Every one
  is inside `reject_degenerate_narrowing`'s tolerance band and would be flagged degenerate.
- The globally feasible ratio is **1.0000**: in every split some class needs the entire pool.

There is no ratio between 0.55 and 1.0 that both narrows meaningfully and stays feasible, so this is
not a tuning problem.

**Why the proxy missed it, stated plainly so it is not repeated.** Two errors compounded:

1. **It measured per assignment.** `required_counts_by_class`
   ([construction_helpers.py:145-155](../experiments/2_benchmark_patch/code/imbalance_benchmark/manifest/construction_helpers.py#L145-L155))
   unions every count across *all* conditions **and all tail assignments**, so the pool is sized by a
   cross-assignment maximum. A class that is mid-pack under `native` can be the head class under
   `difficulty_reversed`'s severe skew, and that interaction is invisible to a single-assignment
   measurement.
2. **It treated `r_min = MIN_POOL_PATIENTS / P_wide` as the achievable ratio.** But `P_wide` is not a
   free quantity that happens to sit above the floor — `_designate_floor` takes
   `pats[:min_independent_units]` and `_expand_pool` prefers adding a slide from a **retained**
   patient over adding a new one, so the designated pool is *already the minimal patient set* that
   supplies `max(counts)` under the caps. The distance from the wide pool to the narrowest feasible
   pool is zero by construction, not by coincidence, and it is zero on all four datasets.

Keep this finding. It is the reason the design changed, and *"the evidence pool cannot be narrowed
because it is already minimal"* is a genuine structural property of the construction.

## Replacement: create the shortage by spreading the reference, not narrowing the treatment

Invert the axis. The existing minimal pool is already the deprived end; what is missing is a
**less** deprived arm to contrast it against. The spread arm distributes the *same* nominal patch
counts over the full eligible patient inventory.

The two directions meet opposite bounds, and only one of them is occupied:

| | narrow (superseded) | spread |
| --- | --- | --- |
| Bound | contribution caps: ≥10 patients, ≥20 slides | eligible patient inventory |
| Distance to it | **zero** by construction | 1.0×–17.1×, measured |
| Per-unit contribution | rises, toward the caps | falls, away from the caps |
| Large `max(counts)` | makes it harder | irrelevant — `max(counts)` exceeds eligible patients by orders of magnitude |

The structural bonus is that the datasets with the *least* narrowing headroom — many patients, so a
pool that is a small fraction of inventory — have the *most* spreading headroom. The axis becomes
available on exactly the multiclass datasets the benchmark needs for RQ3.

## Measured decision (2026-08-29)

Two read-only passes, neither of which wrote to the outputs tree or touched any `pilot`, prepared
manifest, or freeze artifact.

**Inventory scan** — `outputs/independent_support_feasibility/measurement-20260829-r2.json`
(`sha256=3a80d128ae2448d88751c0316c98a2174f41419920e251d8857cda52fac6471a`), retained: eligible
patients and slides, patches-per-patient quantiles, and designated pool support per class and split.

**Construction canary** — the real `designate_patch_pool` call, with the real crossed
`required_counts` rebuilt from each frozen `manifest_freeze.json`, run twice per class: once at the
current `independent_floor` and once with the patient floor raised to that class's full eligible
inventory. This is the check the superseded decision skipped.

Retained at `outputs/independent_support_feasibility/spread-canary-20260829.json`
(`sha256=0842403eae2cfcc3394ce01eb0c4eee14c6fdc477b87cd73d7bdd0a55dcd17c5`), reproducible with
[`spread_pool_canary.py`](../experiments/2_benchmark_patch/code/spread_pool_canary.py). Every class of
every split of every dataset — 12 dataset–split cells, 123 class cells:

| Dataset | Classes | Construction failures | Nesting holds | Classes deprived | Arm mean log-shortage | Per-class range |
| --- | ---: | ---: | --- | ---: | --- | --- |
| BRACS | 7 | **0** | yes | 6 of 7 | 0.41–0.67 | 0.04–1.24 |
| CAMELYON16 | 2 | **0** | yes | 1 of 2 | 1.66–2.53 | 1.66–2.53 |
| TCGA-UT | 30 | **0** | yes | 29 of 30 | 1.46–1.79 | 0.14–2.84 |
| PANDA | 2 | **0** | yes | 1 of 2 | 0.85–0.89 | 0.85–0.89 |

"Nesting holds" means the concentrated pool's patients are a subset of the spread pool's in every
class — the invariant plan 04 asserts, so the contrast adds patients rather than swapping them.

**Every dataset carries the independent axis, in every assignment.** The per-dataset scoping map
therefore becomes a cost knob for plan 08 rather than a feasibility gate, which is the opposite of
the superseded decision, where only BRACS was thought to carry it at full factorial and TCGA-UT and
PANDA not at all.

The ordering inverts for a structural reason worth stating: the datasets with many patients per class
have pools that are a small fraction of inventory and therefore the most room to spread, while being
exactly the datasets that had no room to narrow. TCGA-UT moves from *"no feasible narrowing"* to a
dose of 1.46–1.79 across 29 of 30 classes — 63–78 % of `moderate`'s nominal `log(10) = 2.30`.

`INDEPENDENT_NARROW_RATIO` is deleted. The spread arm takes **no ratio constant**: each class spreads
to its own eligible inventory, which is the natural opposite extreme to a pool that is minimal by
construction, and it removes the free parameter that the superseded decision had to justify.

`reject_degenerate_spreading` gates on the **arm mean achieved log-shortage**, at a floor of
**0.25**. That number is bracketed from both sides by measurement rather than guessed: it sits below
the weakest measured arm (BRACS split 1, 0.409) by a factor of 1.6, and above everything the
superseded narrowing could ever have achieved — its feasible ratios of 0.857–0.968 correspond to
log-shortages of 0.033–0.154. An arm that collapses back toward the narrow arm's reachable range is
rejected; every arm measured here passes.

Do **not** gate per class. One class per split on BRACS and one on TCGA-UT have eligible counts equal
to their concentrated counts and therefore contribute exactly zero; that is expected, and plan 04's
per-axis deprivation set excludes them from the mean rather than dragging it toward zero.

## Work

1. **Measure, read-only.** Do not re-run `pilot`: its `definitive_floor`, quotas and
   `difficulty_evidence` are *inputs* to the new design, and its signed `pilot_report.json` is
   verified at `freeze_execution.py:102` — re-running only invalidates signatures.
2. **Run the construction canary, not a proxy.** Any feasibility claim about pool designation must
   come from `designate_patch_pool` itself, against `required_counts` unioned over every condition
   **and every assignment**. This is non-negotiable now; the proxy is what produced the superseded
   decision.
3. **Decide each dataset's status** from the canary, and record the achieved dose so plan 07 can
   state the dose it is testing rather than discovering it afterwards.
4. **Specify the freeze-time guards** that plan 04 implements: `reject_degenerate_spreading`,
   `reject_non_nested_pools`, and `reject_constant_signal_axes` alongside the existing
   `reject_degenerate_conditions` ([freeze_execution.py:141](../experiments/2_benchmark_patch/code/imbalance_benchmark/commands/freeze_execution.py#L141)).

## Done when

Every dataset is marked as carrying the independent axis or not, on evidence from the real pool
designation rather than a proxy; the achieved dose per dataset is recorded; and the degeneracy
tolerance is chosen from measured numbers.
