# Four-condition LoRA Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure frequency, coverage, interaction, and the conditional recoverable share under the same LoRA recipe as the mitigation experiments.

**Architecture:** Add experiment 48 as a small consumer of exp-40 training/evaluation and the existing allocation/bootstrap helpers. Extend stage-one training with optional fixed CE class weights and opt-in training diagnostics; keep existing callers unchanged. Fit four freshly initialized, matched conditions and analyze their paired predictions.

**Tech Stack:** Existing Python 3.12+, NumPy, PyTorch/PEFT/timm, pandas, pytest, Matplotlib, SLURM/Apptainer, and LaTeX; no new dependency.

**Spec:** `docs/superpowers/specs/2026-11-08-lora-frequency-coverage-design.md`

## Global Constraints

- BRACS and TCGA-UT, Virchow2, imbalance ratio rho=100 only.
- Three existing patient splits, ten allocation draws/split, four arms/draw: 240 fresh fits. Draw 0 of each split on both datasets is the 24-fit pilot; the remaining 216 fits follow technical validation.
- BRACS: 10 patients/class, 2,240 patches, 600 updates. TCGA-UT: 20 patients/class, 19,200 patches, 2,400 updates.
- LoRA rank 8, alpha 16, dropout 0; linear head; batch 256; AdamW learning rate 1e-4, weight decay 1e-4; existing cosine schedule and deterministic image preprocessing.
- Uniform row sampling with replacement in every arm. Fixed loss weights change class influence; B/F share their sampled sequence, as do C/R. Reset initialization for every arm.
- Primary outputs are patient-macro BA and absolute contrasts in percentage points. Share h is secondary, untruncated, conditional on this learner and recipe.
- Preserve repeated-patient dependence across the overlapping test splits. Pilot continuation must not depend on an effect's direction, size, or significance.
- Fresh outputs only; preserve previous experiment artifacts and summary numbers. No automatic push. User-requested commits go to `main`, one per phase, message from `/caveman-commit`; do not create a branch.
- Before editing Python, read clean-code rules; after changes, run that skill with the scope below. Run affected tests only outside that skill.
- Before remote execution, use the project hydra-cluster skill. At most 100 account-wide queued/running tasks; no explicit job time limit; heavy work runs on compute nodes.
- This plan is a handoff, not authorization to submit jobs during the planning turn.

## Review Focus

1. Incorrect batch normalization of weighted CE, especially a batch missing a rare class: fixed weights must retain dataset normalization (Task 1).
2. Initialization depends on execution order or only the sampler is seeded: all four arms must start with the same trainable state within a block (Task 2).
3. An interrupted run, reordered predictions, or changed config is mistaken for a completed compatible fit: reject incompatible records and never pool them (Tasks 2–3).
4. Small/zero damage or shared patients across splits creates a misleading share interval: preserve pairing, expose denominator instability, never drop invalid draws silently (Task 3).
5. Resume submits duplicate tasks or a whole array crosses the account limit: subtract complete/in-flight work and account for every prospective worker (Task 4).

---

## File map and commands

All paths below are repo-relative. These prefixes are literal directory abbreviations for the file lists and commands:

```powershell
$exp48 = 'experiments/03_class_imbalance/01_cause/cross/48_lora_decomposition'
$methods40 = 'experiments/03_class_imbalance/02_mitigation/40_methods'
```

| Files | Responsibility |
| --- | --- |
| Modify `$methods40/code/mitigation/train/stage1.py`, `stage1_step.py` | Optional weighted CE and diagnostics, preserving default training |
| Modify `$methods40/tests/test_fit.py` | Weighted-loss, gradient, microbatch, and compatibility checks |
| Create `$exp48/code/_bootstrap.py`, `__main__.py` | Existing import bootstrap pattern and CLI |
| Create `$exp48/code/lora_decomposition/__init__.py`, `protocol.py` | Package entry and locked conditions/configuration/weights |
| Create `$exp48/code/lora_decomposition/fit.py` | Seeded fits, provenance, record validation, streamed evaluation |
| Create `$exp48/code/lora_decomposition/analyze.py` | Paired contrasts, denominator checks, uncertainty diagnostics |
| Create `$exp48/code/lora_decomposition/submit.py` | Pilot/main sparse arrays using existing SLURM helpers |
| Create `$exp48/code/lora_decomposition/report.py` | Tables/figures from analysis JSON |
| Create `$exp48/configs/bracs.yaml`, `tcga_ut.yaml` | Source recipes plus exp-48 output/SLURM settings |
| Create `$exp48/tests/conftest.py`, `test_protocol.py`, `test_fit.py`, `test_analyze.py`, `test_submit.py` | Focused synthetic tests, no remote/model downloads |
| Create `$exp48/report/48_lora_decomposition.tex`, `references.bib` | Protocol, later results/discussion; generated tables/figures and PDF alongside |
| Modify `experiments/03_class_imbalance/AGENTS.md`, `experiments/AGENTS.md` | Register exp-48 with its actual status, then results when available |

Reuse directly: `mitigation.data.load_shard/train_arm/eval_arm/run_dir`, `run_stage1`, `stream_embeddings/head_logits/evaluate_and_record`, `analyze.bootstrap.selected_ba`, `analyze.paths.fixed_dirs`, `BootstrapContext`, `canonical_permutation`, `decomposition.model.draw_weights`, and `centre.analyze.pooled`. Avoid copying the exp-40 trainer or introducing another results schema. A small experiment-local provenance sidecar supplements the standard run record.

### Task 1: Add fixed-weight CE without changing existing methods

**Files:** The two shared training files and `$methods40/tests/test_fit.py` above.

**Interfaces:** Extend `run_stage1(config, device, arm, method, param, seed, *, class_weights: np.ndarray | None = None, audit: dict[str, Any] | None = None) -> Stage1Output`. Thread an optional `class_weights: torch.Tensor | None = None` keyword through `build_loss_step` and its CE closure. Non-CE methods reject non-None class weights. Default calls retain existing behavior.

- [ ] Add failing `test_weighted_ce_uses_fixed_dataset_normalization`: a two-class dataset with counts `[3,1]`, uniform target, and weights `[2/3,2]`; a batch containing only class 0 must have loss `CE * 2/3`, not `CE`. On the complete four-row dataset, verify the loss/gradient equals the explicitly weighted per-example mean. All-one weights equal ordinary CE.
- [ ] Add `test_weighted_ce_microbatch_matches_full_batch`: one identical update with batch 4 and microbatch 2 produces matching parameters. Test wrong weight length, nonfinite/nonpositive weights, and weights supplied to LA/GCL all raise `ValueError`.
- [ ] Run `uv run pytest "$methods40/tests/test_fit.py" -k 'weighted_ce' -q`; confirm failure concerns the missing weighted-loss feature.
- [ ] Implement fixed float32 class weights and `mean(weights[y] * CE(reduction='none'))`; never use the weight-sum denominator of the built-in weighted-mean CE. Keep sampling unchanged. Do not globally reseed legacy callers.
- [ ] Implement opt-in audit capture in stage one without changing the shared optimizer loop: hash the initial trainable adapter/head tensors before training; wrap the step closure to count visited row indices, stream their digest, and collect finite loss diagnostics. Populate `initialization_sha256`, `sample_sequence_sha256`, `unique_rows_per_class`, `sample_count`, `loss_finite`, `first_loss_mean`, `last_loss_mean`. First/last means cover ten optimizer updates (or all updates for shorter tests), weighted correctly over microbatches. Hash row indices as consecutive fixed-width integers, independent of chunk boundaries. With `audit=None`, skip this work.
- [ ] Add `test_training_audit_matches_observed_batches`: captured sample count equals `steps * batch_size`; per-class unique counts equal the indices actually consumed; first/last means are finite. Exercise a nonfinite loss and require failure before a completed run record can be written.
- [ ] Run `uv run pytest "$methods40/tests/test_fit.py" "$methods40/tests/test_methods.py" -q`; require all tests to pass, including the existing trainable/frozen parameter checks. Defer the implementation-phase commit until Task 4 validation.

### Task 2: Implement the four conditions and reproducible fits

**Files:** `$exp48/code/_bootstrap.py`, `lora_decomposition/{__init__,protocol,fit}.py`, both configs, `tests/{conftest,test_protocol,test_fit}.py`.

**Interfaces:**

- `protocol.ARMS = ('B', 'F', 'C', 'R')`; `load_protocol_config(path: str | Path) -> dict[str, Any]`.
- `arm_allocations(arm: str) -> tuple[str, str]`: B=`('r1','r1')`, F=`('r1','r100')`, C=`('r100','r1')`, R=`('r100','r100')`; tuple means data allocation, loss-target allocation.
- `loss_weights(data_counts: np.ndarray, target_counts: np.ndarray) -> np.ndarray`: `target_counts / target_counts.sum() / (data_counts / data_counts.sum())`.
- `block_seed(split_idx: int, draw_idx: int) -> int`: `48000 + 100 * split_idx + draw_idx`; accept only split 0–2/draw 0–9.
- `fit.run_fit(config: dict[str, Any], split_idx: int, draw_idx: int, arm: str) -> Path`.
- `fit.validate_run(config: dict[str, Any], split_idx: int, draw_idx: int, arm: str) -> bool`: missing/incomplete returns False; incompatible completed provenance raises with the path and mismatch; valid returns True.

- [ ] Bootstrap exp-40 by locating the `03_class_imbalance` ancestor, then its `02_mitigation/40_methods/code`; follow exp-45's `runpy.run_path` pattern. Use the unique package name `lora_decomposition` to avoid its imported `analyze` package. Tests load this bootstrap through their own conftest.
- [ ] Add the failing tests below and run `uv run pytest "$exp48/tests/test_protocol.py" -q`:

```python
def test_four_arm_weight_identities():
    balanced = np.array([2, 2])
    skewed = np.array([3, 1])
    assert arm_allocations('F') == ('r1', 'r100')
    assert arm_allocations('C') == ('r100', 'r1')
    np.testing.assert_allclose(loss_weights(balanced, balanced), [1, 1])
    np.testing.assert_allclose(loss_weights(skewed, skewed), [1, 1])
    np.testing.assert_allclose(loss_weights(balanced, skewed), [1.5, 0.5])
    np.testing.assert_allclose(loss_weights(skewed, balanced), [2/3, 2])

def test_block_seeds_are_distinct():
    assert len({block_seed(s, d) for s in range(3) for d in range(10)}) == 30
```

- [ ] Implement protocol/config loading: deep-copy source settings from exp-41 BRACS or exp-43 TCGA-UT, replace outputs and SLURM code paths, remove mitigation candidate grids/stage-two execution, and validate the exact recipe in Global Constraints. Reject zero/negative/mismatched class counts, unknown arms, altered rho, unsupported encoders, and output paths resolving outside exp-48. Tests assert source config is unchanged and both workload counts are 120.
- [ ] Use `train_arm` twice to obtain r1/r100 rows and realized counts for each block. Select each arm's data/target counts by `arm_allocations`; verify equal total budgets and the expected per-dataset counts. Seed Python, NumPy, Torch CPU and CUDA before each fresh `run_stage1` call. Pass the same block seed and `method='ce', param=0.0`, with the appropriate weights and a new audit dictionary. Do not reuse a trained encoder or head between arms.
- [ ] Stream validation/test embeddings with existing helpers and write the standard record with `RunMeta(config, 'ce', None, arm, class_counts)`. Use `run_dir(shard, arm, 'ce', None, draw_idx)` so `fixed_dirs` can load it later. Do not produce stage-two caches.
- [ ] Add `provenance.json` as the completion marker, written atomically after the standard record and arrays pass validation. Record arm/split/draw/seed, ordered training identity digest, counts/target shares/weights, ordered validation/test identity digests, recipe/source-code/input-manifest fingerprints, pinned encoder revision/weight hash, software/device details, and training audit. Source-code fingerprints cover the exp-48/shared files actually used, not unrelated repository changes. Verify labels, class order, array length/shape, finite logits, and identities before accepting a record. Never overwrite an incompatible completed run automatically.
- [ ] Add a tiny-encoder test that fits all four arms in reverse and forward execution order. Assert initialization digests agree across arms; B/F and C/R sampling digests and unique exposure agree; seed changes across blocks; predictions reproduce within the supported numerical tolerance. Check B/R all-one-weight training matches unweighted CE under identical seeds. Add resume tests for missing sidecar/arrays, changed recipe, and reordered prediction identity. Use synthetic models/images, no pretrained downloads.
- [ ] Run `uv run pytest "$exp48/tests/test_protocol.py" "$exp48/tests/test_fit.py" -q`; require pass. Document numerical reproducibility scope (fixed software/hardware; no cross-device bitwise promise).

### Task 3: Implement paired analysis and honest share intervals

**Files:** `$exp48/code/lora_decomposition/analyze.py`, `tests/test_analyze.py`.

**Interfaces:** `contrasts(ba: dict[str, np.ndarray]) -> dict[str, np.ndarray]`; `summarize(distribution: np.ndarray) -> dict[str, float]`; `analyze_dataset(config: dict[str, Any]) -> dict[str, Any]`; `run_analyze(configs: list[dict[str, Any]]) -> Path` writes `$exp48/report/analysis.json` for exactly both datasets. Arrays use observed estimate at index 0, bootstrap replicates afterward.

- [ ] Add a failing algebra test with B=70, F=68, C=69, R=66: assert D=4, D_F=2, D_C=1, I=1, G=3, h=0.75, and `G == D_F + I`. Add a valid h>1 example and assert it remains untruncated. Check `summarize([999, 1, 2, 3, 4])` uses 999 only as the point estimate, not in its quantiles. Run `uv run pytest "$exp48/tests/test_analyze.py" -q` to confirm meaningful failure.
- [ ] Require all 120 compatible records per dataset via Task 2. Reject partially observed blocks rather than silently averaging available arms. Build one shared crossed-patient bootstrap per dataset with 10,000 replicates and existing `BOOTSTRAP_SEED`; share draw weights across all arms. The replicate count is a reproducibility choice, not a claim of increased statistical precision. Use `selected_ba(fixed_dirs(...), ...)` and `pooled` with equal split weights and ten draws/split.
- [ ] Implement positive damage conventions from the spec exactly. Produce full paired intervals, equal-draw patient-only intervals, observed-patient draw-only intervals, and per-split estimates. Derive diagnostics from the same per-fit arrays: patient-only uses all-one draw weights; draw-only broadcasts column 0 over the shared draw resamples. Do not sum these widths or call draw-only variability pure allocation noise.
- [ ] Implement share status: if D point <1 pp or its 95% interval is not strictly positive, set `usable_for_hypothesis=false`. Store nonpositive/zero denominator counts and an explicit reason. For finite nonzero denominators, retain all signed ratios. If an exactly zero denominator makes a ratio nonfinite, keep the absolute contrasts, mark ratio interval unavailable, and serialize null/status rather than invalid JSON Infinity/NaN. Never compute a cleaned finite-only interval or divide by a fixed observed denominator.
- [ ] Add tests with identical patient-level arm predictions (paired difference identically zero), a patient repeated in two split manifests (same bootstrap weight), a missing arm, and denominator samples crossing/at zero. Assert the patient-only/draw-only diagnostic construction above and no inclusion of the observed column in interval quantiles. Report 95% intervals as marginal estimation intervals, not a simultaneous significance screen over every component.
- [ ] Run `uv run pytest "$exp48/tests/test_analyze.py" -q`; require pass. JSON includes protocol/fingerprint identifiers, sample/replicate counts, all four accuracies, contrasts, share status, per-split results, diagnostics, and the fixed-recipe/conditional interpretation.

### Task 4: Wire the CLI, safe sparse submission, and pilot checks

**Files:** `$exp48/code/__main__.py`, `lora_decomposition/submit.py`, `tests/test_submit.py`; complete both configs.

**Interfaces:** CLI subcommands `fit --shard-index 0..29 --arm B|F|C|R`, `check --phase pilot|main`, `analyze`, `submit --phase pilot|main [--dry-run]`. Repeat `--config` only for `check`/`analyze`; fit/submit require one dataset. `submit_jobs(config: dict[str, Any], phase: str, dry_run: bool) -> list[str]`; `check_runs(configs: list[dict[str, Any]], phase: str) -> Path` writes a check JSON with explicit pass/fail and reasons.

- [ ] Add mocked submission tests: pilot shard indices are `{0,10,20}` for each arm; main indices are the other 27; validation rejects out-of-range indices. Completed and matching in-flight tuples are excluded. With 98 account-wide tasks, never submit a three-worker array. Empty pending work submits nothing. `--dry-run` cannot call `sbatch` or require working SSH. Run `uv run pytest "$exp48/tests/test_submit.py" -q` and observe missing-feature failures.
- [ ] Reuse `build_job`, `SlurmJob.array_indices`, the renderer and queue helpers used by `breadth.slurm`. Each task runs one arm, not a four-arm bundle. Keep stage name `fit` so SquashFS binds remain active. Use array/job identifiers in logs and retain submitted IDs plus arm/shard mapping. Do not rely on `check_queue_cap` alone: it checks current occupancy but does not reserve all prospective array workers. Restrict each submission to available slots, recheck between submissions, and leave remaining work pending for a later invocation.
- [ ] Use BRACS `gpu-2h`, TCGA-UT `gpu-5h`, one GPU, 8 CPUs/64 GB and the source dataset mount/constraint settings. Copy verified source paths for the container and benchmark manifests; no explicit `--time`. Render a GPU-test smoke job separately before full pilot work; smoke uses a temporary config and an isolated root under `$exp48/outputs/smoke/`, excluded from scientific results. Its shortened training budget is an explicit smoke-only override that cannot pass scientific run validation. Analysis uses the source CPU resource settings on compute nodes.
- [ ] `check --phase pilot` validates all 24 pilot fits, audit equality for each block, manifests/arrays, finite training/validation/test outputs, and recipe consistency. Its machine result contains no effect-size/significance criterion. Keep a short human review of training/validation diagnostics in `report/pilot_check.md`; record any optimization concern without tuning on test outcomes. Main submission requires the technical pilot check and documented review; no new approval requirement is imposed if execution was already authorized.
- [ ] Run `uv run pytest "$exp48/tests" "$methods40/tests/test_fit.py" "$methods40/tests/test_methods.py" "$methods40/tests/test_cli.py" -q`. Then run the clean-code skill (allow at least ten minutes, poll for completion):

```powershell
uv run python 'C:/Users/Yannik/.codex/skills/clean-code/run.py' --scope experiments/03_class_imbalance --vulture-scope "$exp48/code,$methods40/code"
git diff --check
```

Expected: affected tests pass, clean-code exit 0, no whitespace errors. Fix violations without relaxing rules. No separate full-suite pytest invocation.

- [ ] When the user requests the implementation-phase commit, stage only exp-48, the shared files/tests above, and the approved design/plan. Use `/caveman-commit` for an `exp48` Conventional Commit on `main`; do not push unless asked.

### Task 5: Write the protocol, execute in stages, and report results

**Files:** `$exp48/code/lora_decomposition/report.py`, report files, `tests/test_analyze.py`, and the two experiment index files above. Later generated outputs: `report/analysis.json`, `arm_table.tex`, `component_figure.pdf`, `recovery_figure.pdf`, `results.tex`, and compiled report PDF.

**Interface:** `render_report(analysis_path: Path, report_dir: Path) -> None`, called explicitly after `run_analyze`; it renders supplied estimates and never refits or reselects methods.

- [ ] Use the scientific-writing skill to write the protocol before any scientific fit: motivation, controlled four-arm table, weighting equation, fixed budgets, sampling/initialization pairing, positive-damage estimands, inference, and conditional hypothesis. Register exp-48 as planned/in progress in the indexes; do not insert results prematurely. Cite the original decomposition and mitigation protocols using their existing bibliography entries.
- [ ] Add a synthetic report test using the Task 3 fixture: generated table contains D=4 and h=0.75; unstable share produces an explicit unavailable label and no numeric ceiling band. Implement figure/table generation from JSON only. Plot signed component intervals rather than stacking away a negative interaction; label h as the recovery benchmark. Keep historical mitigation comparisons descriptive and distinguish recovery from balanced-gain-corrected recovery.
- [ ] Compile the protocol from Git Bash: `bash scripts/tex-build.sh experiments/03_class_imbalance/01_cause/cross/48_lora_decomposition/report/48_lora_decomposition.tex`. Inspect warnings, render/inspect all PDF pages using the PDF skill, and run `git diff --check`.
- [ ] Once execution is authorized, inspect remote code/input state and working noninteractive SSH using the hydra-cluster skill. Verify container/checkpoint/dataset paths and manifests, sync only intended changes without overwriting remote edits, and render/inspect jobs before submission. Run the compute-node smoke, then the pilot. From the project root on the login node, the planned CLI is:

```bash
.venv/bin/python experiments/03_class_imbalance/01_cause/cross/48_lora_decomposition/code --config experiments/03_class_imbalance/01_cause/cross/48_lora_decomposition/configs/bracs.yaml submit --phase pilot --dry-run
```

Run the same for TCGA-UT; after inspecting both, submit with `--dry-run` removed. Monitor through the cluster skill's wait helper, reconcile with `sacct` and artifacts, and run `check --phase pilot` for both configs. Actual full scientific workload remains 24 pilot +216 main fits; smoke is separate.

- [ ] If technical/pilot review passes, submit the main phase in available queue capacity. Do not refit valid pilot blocks. A changed scientific recipe requires a new fingerprint and consistent refitting of affected conditions, not mixing runs. Resume only missing, non-in-flight work. After all 240 complete, run `check --phase main` (checks all 30 blocks/dataset) and run `analyze` with both configs on a compute node.
- [ ] Render the results, inspect the full/patient-only/draw-only uncertainty, and state whether the LoRA benchmark differs from the old frozen-probe reference without claiming a causal LoRA effect from the different fitting procedures. Retain the fixed-split/one-seed-per-draw limitations. If shares remain noisy or unusable, report that outcome and the absolute contrasts; do not expand the experiment opportunistically.
- [ ] Compile and visually inspect the final report, run affected report tests and clean-code for any subsequent Python edits, and run `git diff --check`. Update indexes with verified results. Leave `summary/ci_summary.tex` unchanged pending result review. On a requested results-phase commit, use `/caveman-commit`, scope `exp48`, only report/index files; no automatic push.

## Completion and execution handoff

Implementation is ready when Tasks 1–4 tests/clean-code pass, the protocol compiles and is visually checked, and dry-run jobs describe the approved workload. Scientific completion additionally requires all 240 compatible fits, validated paired analysis, and the checked report. These are separate completion claims.

Recommend native execution with `superpowers:executing-plans`: the training, provenance, and analysis tasks share tight interfaces, and this is a small extension of one existing pipeline. A final independent review should focus on weighting, pairing, provenance, and denominator behavior before expensive fits. The user can instead select subagent-driven execution under the corresponding skill.

Self-review completed: scope, interfaces, tests for the five review risks, pilot/main counts, inference, reporting, and repository checks are covered. No implementation, tests, remote checks, or submissions were performed while writing this plan.
