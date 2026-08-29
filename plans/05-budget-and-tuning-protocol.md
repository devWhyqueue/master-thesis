# 05 — Budget and tuning protocol

**Depends on:** 01. **Gates:** 07. **Cluster cost:** none directly; changes 07's and 08's cost.
Independent of 04 — can proceed in parallel.

**Must land before the BRACS run.** The budget definition changes which configs win, so a BRACS run
started before this lands would have to be retuned from scratch. BRACS is also where every claim in
this plan gets its first test: it ran all fifteen roster methods, and cRT, OKO, MDE and the two soft
hybrids are exactly the methods whose exposure was unmatched.

## Context

The exploratory roster is confounded on two axes at once, and the Discussion's "no unmet need"
conclusion rests entirely on it. Fixing this is expected to *change* that conclusion, not confirm it.

## 1. The budget confound

Budget is `U = 30 * ceil(T/B)` optimizer **updates**
([context.py:44](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/context.py#L44),
[training/__init__.py:86-88](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/__init__.py#L86-L88)).
But methods consume wildly different amounts of data per update:

| method | examples per update | source |
| --- | --- | --- |
| every single-loader patch/MIL method | `B` | `training/__init__.py:122` |
| `oko` | `B*(k+2)`, k up to 8 | [oko/__init__.py:140](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/oko/__init__.py#L140) |
| `mde` | `2B` | `special_methods.py:114-133` |
| `crt` | two stages in ratio 1 : 0.2 | `multistage.py:78,88` |
| `post_hoc_logit_adjustment` | 0 (reuses CE's checkpoint) | `confirmation.py:85-112` |

So OKO saw **3–10× the data** of every other method at equal U. That, not any property of
"batch-level mechanisms", is the plain explanation for its 1.646 / 1.784 TCGA-UT overshoot past the
balanced reference. The cost appendix cannot arbitrate because `cost.tex` covers only the five
confirmatory members.

**Fix.** Replace the update budget with an example-presentation budget:

> `E = REFERENCE_PASSES * T`, then `U_method = ceil(E / examples_per_update(method, B, cfg))`

Accept two consequences explicitly:

- **OKO's budget becomes config-dependent**, since `k` comes from the candidate config. That is
  correct under exposure matching — it is what makes OKO's own tuning comparison honest — but the
  budget can no longer be a frozen scalar per condition. **Freeze `E`; derive `U` at fit time.**
- The soft hybrids' forced sampler does not change exposure (same `B` per update), so it needs no
  budget change. It remains an untuned design asymmetry — see plan 01's B2 and item 3 below.

**Where the frozen record changes:** `update_budget` → `example_budget` +
`updates_for_exposure(...)` in `training/__init__.py:86-88`; `"update_budgets"` →
`"exposure_budgets"` at `freezing.py:238-244` (changes `content_sha256`, but a full re-freeze is
happening anyway); `resolve_update_budget` at `context.py:233-236` gains `method` and `cfg`; its five
call sites; the six consumers of `freeze["update_budgets"]`; `run_context.updates_for`;
`training/config.py:103-111` records `"budget_unit": "example_presentations"`; and
`commands/smoke.py:64-65`'s two `REFERENCE_PASSES = 2` monkeypatch points must stay valid.

## 2. Checkpoint interval must be rescaled, or the fix creates a new confound

`resolve_checkpoint_interval`
([config.py:67-76](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/config.py#L67-L76))
is `max(configured, ceil(budget / TARGET_CHECKPOINTS))` in **updates**, floored at
`CHECKPOINT_INTERVAL = 50`. Under exposure matching, U varies by up to 10× across methods, so the
fixed floor would give OKO roughly ten times fewer checkpoints than CE at equal exposure — **a new
confound in model selection, replacing the one being removed.**

Make `TARGET_CHECKPOINTS = 170` the invariant and stop letting the floor dominate:
`interval = max(1, round(budget / TARGET_CHECKPOINTS))`. Every method then gets ~170 validation passes
regardless of its U. Total validation cost is flat — the same *number* of evaluations, just more
frequent in update terms for OKO, MDE and cRT.

## 3. Mechanism disclosure in the roster

`ce_soft_f1` and `ce_soft_mcc` force a balanced sampler at strength 1.0, untuned
([loaders.py:28,59-60](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/loaders.py#L28-L60)),
and cRT's stage two is hard-coded to `balanced_sampling` at 1.0 (`multistage.py:82-88`). So balanced
sampling, soft-F1, soft-MCC and cRT stage two are **four dressings of one mechanism**. The claim that
"no gate-passing unit is beyond repair, some tested mechanism reaches above 0.67" rests on counting
them as independent evidence.

Report them as sampling+loss hybrids, and extend the cost appendix to the **whole** roster rather
than the five confirmatory members, so the cost argument the Discussion wants to make is actually
supported by data.

## 4. Tuning protocol asymmetries

- **Objective is balanced accuracy only.** `_selection_key`
  ([tuning_aggregate.py:62-70](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/workflows/tuning_aggregate.py#L62-L70))
  selects on natural-validation balanced accuracy, tie-broken by macro-F1 then NLL. Tail-group NLL is
  a **co-primary** endpoint with its own gate, yet no method was ever tuned for it. Every "this method
  degrades probability quality" claim (15 of 18 cells) is about methods optimized for a different
  endpoint. Either select per gate axis, or state the asymmetry in the protocol as a limit on that
  claim.
- **Selection pools across tail assignments.** One configuration per (method, dataset, severity),
  averaged over native/aligned/reversed. The assignment determines *which classes need help*, so this
  systematically dilutes exactly the difficulty and diversity methods that "recover nothing". Either
  select per assignment or justify the pooling.
- **98 axes ended tuning-limited at an envelope bound**, concentrated in balanced sampling (14),
  soft-F1 (13), soft-MCC (11), prevalence (10). The learning-rate envelope tops out at 1e-2
  ([search_windows.py:19](../experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/workflows/tuning/search_windows.py#L19))
  and CAMELYON16 pins there repeatedly. A reported optimum at a bound may not be one — widen the
  envelope or report the count prominently.
- **Tuning budget differs per method**: 16 candidates for most, 4 for CE and cRT, 4 for OKO on binary
  targets (`k <= K-1`), 1 shard for post-hoc LA (`tuning_schedule.py:130-135`). Defensible, but it
  should be stated rather than left to be discovered.

## Verification

Breaking tests: `test_update_budget_formula` (`tests/modeling/test_training.py:179`) and
`test_training_uses_the_frozen_update_budget` (`:202,272`) both pin the update budget;
`test_resolve_checkpoint_interval_scales_with_budget` (`:357`) has three hard-coded values that all
change; cost-payload arithmetic in `tests/analysis/test_metrics.py:252,339`,
`tests/modeling/test_tuning_execution.py:160`, `tests/modeling/test_tuning_shards.py:293`.

The one new test that states the whole contract:

```python
@pytest.mark.parametrize("method,cfg",
    [("ce", {}), ("oko", {"parameter": 8}), ("mde", {}), ("crt", {})])
def test_every_method_sees_the_same_example_exposure(method, cfg):
    assert ctx["processed_examples"] == pytest.approx(EXPOSURE, rel=0.05)
```

`processed_examples` is already recorded exactly by every training path, so this asserts directly on
the quantity the budget is meant to equalize — no new instrumentation.

## Done when

Exposure is equal across methods to within 5 %, every method gets ~170 checkpoints, the cost appendix
covers the whole roster, and the four tuning asymmetries are each either fixed or written into the
protocol as a stated limit.

**Expect the headline to move.** OKO's U drops 3–10×, so its overshoot will shrink or vanish. That may
well *create* the unmet-need regime the current report says does not exist. Plan for the Discussion to
be rewritten, not patched.

**Where wave 1 tests this.** The overshoot that motivated the fix was on TCGA-UT (1.646, 1.784), and
TCGA-UT is in wave 1, so the headline symptom itself is directly testable — read it there. BRACS
tests the mechanism rather than the symptom: OKO recovered 0.357 to 0.811 there and never overshot,
so what BRACS shows is whether OKO, cRT and MDE still recover comparably once their exposure matches
everyone else's, and whether the ~170-checkpoint invariant holds across methods whose U now differs
by up to 10×. If BRACS shows those three collapsing under matched exposure, the roster ceiling claim
is already dead before TCGA-UT confirms it.
