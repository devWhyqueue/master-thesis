# Plan 2 — Stop the natural condition burning 35–60% of the tuning budget

**Scope:** one function signature and three call sites. No estimand change, no re-analysis.
**Depends on:** nothing. **Should land before:** Plan 3's TCGA-UT re-run, which is where the
saving is banked.

## Context

The natural condition is CE-only and descriptive — it never enters `D_M` or `R_M`
(protocol §"From imbalance deficit to mitigation recovery"). Its only job is placing controlled
results on the scale of the full training partition and the dataset's own prevalence
(§"Interpretation and Limitations"). Yet it dominates the tuning bill on two datasets:

| dataset | natural | balanced | moderate | severe | natural share |
|---|---|---|---|---|---|
| BRACS | **156.8** | 16.0 | 41.8 | 46.0 | **60%** |
| CAMELYON16 | **116.2** | 55.8 | 80.6 | 79.8 | **35%** |
| TCGA-UT | 38.2 | 99.1 | 317.2 | 308.6 | 5% |
| PANDA | 41.7 | 682.0 | 619.0 | 692.6 | 2% |

GPU-hours for the tuning search, summed over three splits, from
`split=*/data/tuning_search_cost_<condition>.json`. Natural tunes **one** method; the controlled
conditions tune the full eleven-method roster. Per method, BRACS natural costs ~50× a controlled
fit.

## Diagnosis: an unscaled validation cadence, not an expensive condition

`configs/panda_patch.yaml` already records that validation is ~98% of tune cost — a full
validation pass every `checkpoint_interval` steps — and scales the interval to compensate. But
[resolve_checkpoint_interval](experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/config.py#L62-L65)
reads one global `patch_training.checkpoint_interval`, so PANDA's hand-tuned 1500 applies to
every PANDA condition and every other dataset keeps the default 50 — while natural's update
budget is 12–62× the controlled one:

| dataset | controlled U | natural U | ratio | validation passes at interval 50 |
|---|---|---|---|---|
| CAMELYON16 | 8,490 | 523,830 | **61.7×** | 170 controlled vs **10,477** natural |
| TCGA-UT | 16,260 | 262,590 | 16.1× | 325 vs 5,252 |
| BRACS | 3,960 | 50,970 | 12.9× | 79 vs 1,019 |
| PANDA | 257,790 | 739,110 | 2.9× | controlled already scaled to 1500 |

(`update_budgets` from each `manifest_freeze.json`.)

Natural is running 12–62× more full validation passes than the controlled conditions it is meant
to contextualise, for no methodological reason.

## Change

Derive the interval from the update budget instead of pinning it to a constant. Target a fixed
**number** of checkpoints rather than a fixed interval:

```python
TARGET_CHECKPOINTS = 170

def resolve_checkpoint_interval(cfg, is_mil, budget):
    """Validation cadence scaled so any budget takes ~TARGET_CHECKPOINTS passes.

    Validation is ~98% of tune cost, so a fixed interval makes cost scale with U.
    A configured value still wins when it is coarser; this only raises the floor.
    """
    k = "wsi_training" if is_mil else "patch_training"
    configured = cfg.get(k, {}).get("checkpoint_interval", CHECKPOINT_INTERVAL)
    return max(configured, math.ceil(budget / TARGET_CHECKPOINTS))
```

`budget` is already a local at every call site, so nothing needs threading through the workflow:

- [training/__init__.py:145](experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/__init__.py#L145) — `max_steps` in scope
- [special_methods.py:111](experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/special_methods.py#L111) — `budget` is a parameter
- [oko/__init__.py:100](experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/oko/__init__.py#L100) — `budget` is a parameter
- [training/config.py:91](experiments/2_benchmark_patch/code/imbalance_benchmark/modeling/training/config.py#L91) — `resolve_training_config` is provenance-only; record the rule
  (`target_checkpoints`) rather than a single resolved number, since it now varies by condition.

**Sanity check the framing already passes:** PANDA's controlled budget 257,790 / 170 = 1,516,
against the 1,500 that was derived by hand and written into `panda_patch.yaml`. The rule
reproduces the value someone already reasoned their way to.

Expected effect: CAMELYON16 natural ~116 → ~2 GPU-h, BRACS ~157 → ~12, TCGA-UT ~38 → ~2.4.
PANDA is unchanged (its `max` keeps 1500 for controlled; natural moves 50 → 4,348).

Once this lands, the PANDA-specific `checkpoint_interval: 1500` and its explanatory comment can
be deleted — the rule subsumes it. Do that in the same change so there is one mechanism, not two.

## Why not delete the natural condition

It is the only thing placing controlled results on the scale of the complete training partition,
which §"Interpretation and Limitations" relies on to bound the claims. Deleting it saves the same
GPU-hours and loses that. After this fix it costs ~2% of tuning instead of 60%, so the question
does not arise.

## Verification

One runnable check, in `tests/modeling/` next to the closest existing training test:

```python
assert resolve_checkpoint_interval({}, False, budget=8_490) == 50      # small budget: default holds
assert resolve_checkpoint_interval({}, False, budget=523_830) == 3_082 # large budget: scales
# a configured coarser value still wins
cfg = {"patch_training": {"checkpoint_interval": 1500}}
assert resolve_checkpoint_interval(cfg, False, budget=257_790) == 1516
```

Then the gate:

```powershell
uv run pytest experiments/2_benchmark_patch/code/tests/modeling
uv run python "$env:USERPROFILE\.codex\skills\clean-code\run.py" --scope experiments/2_benchmark_patch/code --vulture-scope experiments/2_benchmark_patch/code
```

**On the cluster:** the effect is visible in the next `tuning_search_cost_natural.json` —
TCGA-UT's should drop from 38.2 GPU-h to a few. Nothing to verify for BRACS/CAMELYON16/PANDA in
this cycle; they keep their existing selections.

## Risks

- **This changes selected hyperparameters, not only cost.** Fewer checkpoints means a coarser
  early-stopping grid, so a re-run natural CE anchor may select a different step than it would
  have. Acceptable for a descriptive anchor, but state it rather than presenting a free saving.
- **`TARGET_CHECKPOINTS = 170` is a choice, not a derivation.** It is the number the controlled
  runs already take. Put it next to `CHECKPOINT_INTERVAL` with that one-line justification so the
  next reader does not have to reconstruct it.
- **Do not retro-apply to completed runs.** BRACS, CAMELYON16 and PANDA keep the selections they
  already have; mixing cadences within one dataset's reported results would be worse than the
  cost.
