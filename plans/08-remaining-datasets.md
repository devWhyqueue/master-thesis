# 08 — Wave 2: the two binary datasets

**Depends on:** plan 07's Gate B. **Cluster cost:** the expensive part of the whole programme.
**Conditional — this plan may be closed unopened.**

## What wave 2 can and cannot add

CAMELYON16 and PANDA are binary-target datasets, and `_alignment_identifiable`
([rq3_wiring.py:61-70](../experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/predictors/rq3_wiring.py#L61-L70))
excludes binary cells from both RQ3 fits, because a Pearson correlation over two points is always ±1.
**Neither dataset can contribute to RQ3**, however strong its independent dose. That is a design
consequence of plan 02's identifiability guard, not a scoping preference, and it sets what this plan
is for:

| Contribution | Available here? |
| --- | --- |
| RQ1 deficits and gate routing on a second target type | yes |
| RQ2 five-family separation, replication across target types | yes |
| The "few classes concentrate the damage" contrast (PANDA vs TCGA-UT at equal nominal rho) | yes — needs PANDA specifically |
| A second *independent-axis* dataset at high dose (CAMELYON16, 1.66–2.53) | yes, for RQ1/RQ2 only |
| RQ3 groups, `sigma_u`, leave-one-group-out | **no** |

## Scope is set by plan 07's outcome, not decided here

**If (a) — the independent axis is real and independent-support weighting behaves differently on it:**
run both. CAMELYON16 first: it carries the highest measured independent dose of the four
(1.66–2.53) and costs 8,490 updates, so it is the cheapest replication of the wave-1 finding on a
different target type. PANDA last.

**If (b) — the axis is real but the signal still carries no recovery:** run CAMELYON16 only. It tests
whether the wave-1 separation replicates at a *higher* independent dose, which is the one thing
wave 1 cannot rule out. Skip PANDA: it costs 257,790 updates against CAMELYON16's 8,490, roughly 30×,
and its unique contribution is a single descriptive contrast.

**If (c) — the independent contrast opened no gate on either wave-1 dataset:** close this plan for the
independent axis. CAMELYON16's higher dose is not a counter-argument: it sits at a third of BRACS's
nominal sensitivity, so a null there would be uninformative. Run PANDA alone **only** if the
few-classes contrast is judged worth 257,790 updates on its own merits, which it probably is not.

## Order, when it runs

Per dataset, independently — there is no cross-dataset dependency until `match`:

1. `freeze` (3-split array, `cpu-2d`/`cpu-7d`, non-checkpointable per split).
2. `signals` → the same Gate A checks as plan 07, scoped to the spread arm.
3. `tune` → gate on `tuning_locked` for every condition.
4. `confirm` → `analyze`.

Then, once every included dataset has landed `analyze`:

5. **`match`** — single job, needs every included root's signals. Pooled standardization now runs over
   a *larger* unit population, so wave 1's standardized scores **shift**. Wave 1's labels are
   provisional and must be regenerated here, not carried forward. This is the defect that cost a
   6-hour re-run last time: `match` writes to each dataset root and `analyze` reads from there, so
   **run `match` for every root before re-running `analyze`**, not after.
6. **`combine-rq3`** — still fitted on the wave-1 groups only; wave 2 adds cells that the
   identifiability guard drops. Verify that it does drop them rather than silently saturating the
   alignment column.
7. **`report-tables`**, then rewrite the results report from wave-1 scope to full scope.

## Reuse versus re-run

| Stage | Action |
| --- | --- |
| `prepare` (+ `materialize`/`extract` for PANDA) | **reuse** — plans 01–06 touch no adapter, tiling, feature extraction, split assignment or `manifest.csv`. `verify_prepared_feature_provenance` must still pass; do not delete manifests or feature banks. |
| `pilot` | **reuse** — its signed `pilot_report.json` is an *input*, verified at `freeze_execution.py:102`. Plan 03's measurements were read-only and did not touch it. |
| everything from `freeze` onward | **from scratch** — new conditions, new pools, new budget record, changed `CONDITION_RHOS`, hence a new `content_sha256`. |

**Do not use `amend-grids` or `refreeze-preflight`.** Both preserve the supersession chain for the
case where construction did *not* change. Here it did, so the chain must break: old shards and run
records must be rejected. `accepted_freeze_hashes` does that automatically **provided the new freeze
omits the old hashes from `supersedes`**. Archive `split=*/data/tuning_shards`, `split=*/results` and
the old `manifest_freeze.json` first, then delete them from the working root so nothing stale can be
silently accepted.

## Checkpointing boundaries

**Non-checkpointable — a single job must finish or the work is lost:** `freeze` per split
(`cap_feasible_shared_total`'s probe loop writes no intermediate state; if a dataset times out
repeatedly, memoise `_total_cap_feasible` to a scratch JSON keyed by freeze-input hash and total);
`tune-base-reduce`; `tune-final-reduce`; `match`; `analyze-combine`; `combine-rq3`. `analyze` and
`write_interval_tables` have no checkpointing either — on TIMEOUT go **straight to the longest
partition** rather than retrying the short one. The last wave's `analyze-combine` exhausted `cpu-5h`
at 110 of 113 interval tables; budget `cpu-2d` for it with five conditions.

**Checkpointable, safe to resubmit at array-index granularity:** `tune-shard` (each candidate writes
its own signed artefact; `resume_plan` recomputes exactly the missing indices), `confirm-shard`
(`resolve_confirm_bundle` re-runs only missing `run.json` dirs), `signals`, `prepare-extract-shard`.

**Two standing hazards from prior runs on this pipeline:**

- `resume_plan`'s "remaining" reflects disk state only. Cross-check `squeue` before resubmitting or
  you create racing duplicate jobs.
- The array-index-to-work-item mapping depends on `shards-per-task`. Resubmitting a straggler with a
  value changed since the array was submitted silently drops real candidates instead of failing at
  the SLURM level.

**Queue discipline:** keep queued + running jobs at or under 100 per account. With 5 controlled
conditions the base tuning array is roughly 5 × 196 shards per dataset; raise `tune_shards_per_task`
proportionally rather than flooding the queue.

## PANDA specifically

PANDA is 257,790 updates per run against BRACS's 3,960, roughly 65×. It carries the weakest spread
dose of the four (0.85–0.89), contributes one binary target that RQ3 discards, and was the dataset the
previous RQ3 leave-one-group-out could not place (held-out RMSE 0.0861, against TCGA-UT's 0.0131).
Its one unique contribution is the few-classes contrast against TCGA-UT. It is the last thing to run
and the first thing to cut.

**If PANDA runs, scope its spread arm to `native` only** — it has one assignment anyway, so this
costs nothing, but the `SPREAD_ASSIGNMENTS_BY_DATASET` knob exists for exactly this and should be
used deliberately rather than left at the default.

## Done when

Every included dataset completes through `analyze`, `match` and `combine-rq3` run over the full
included set, and the results report is rewritten from wave-1 scope to the realized scope — stating
plainly which datasets were run, which were not, and that RQ3 rests on the two multiclass datasets
alone regardless of how many were run.
