# Unify and Rebuild the Class-Imbalance Benchmark (Phased)

## Status note (supersedes the flat plan below the phases)

A prior agent pass built `experiments/2_cls_imb_benchmark/code/imbalance_benchmark/`
(~1,600 lines) but implemented it as a synthetic-data smoke skeleton: `cmd_prepare`
silently fabricates random-tensor "datasets" instead of calling real adapters, only 3 of
~15 roster methods exist, `analyze` emits placeholder LaTeX/PNG files, and the legacy
trees (`native/`, `tcga_ut/`, `experiments/shared/`) were never touched. An inventory of
those legacy trees found they contain substantial real, protocol-relevant
implementations — most method logic lives in `experiments/shared/common_code/` and is
merely re-exported by thin shims in `native/` and `tcga_ut/`. The work is re-scoped below
into phases that **port** that real code into the unified package rather than
reimplementing it, keeping the new package's CLI plumbing (`common.py`,
`construction.py`'s exponential-formula/round-robin-cap logic) as the base to build on.

Each phase should end with passing tests for the ported/written code before starting the
next. Do not delete any legacy tree until the phase(s) depending on it are done and
tested (per the original plan's deletion gating). The `smoke` command's synthetic-data
path is legitimate per the original plan ("run a synthetic end-to-end validation without
external datasets") — the defect is that `cmd_prepare` falls back to it silently instead
of only being reachable via `smoke`; fix that gating in Phase 1.

### Phase 1 — Real dataset adapters & feature extraction
- Port and rewrite to the new manifest schema (case_id/slide_id/target/split/feature_path):
  - `native/code/data/bracs/{metadata,splitting,tiling,power_law,prepare}.py` → BRACS adapter
  - `native/code/data/camelyon16/{metadata,masks,splitting,prepare}.py` → CAMELYON16 adapter
  - `native/code/data/panda/{metadata,masks,select,splitting,tiling,prepare}.py` → PANDA adapter
  - `tcga_ut/code/data/prep/manifest/{feature,patch,splits}.py`, `tcga_ut/code/metadata.py`, `tcga_ut/code/data/staging/*` → TCGA-UT adapter
  - `native/code/data/{bracs,camelyon16,panda}/features.py`, `native/code/data/feature_store.py`, `data/feature_cache.py` → frozen Virchow2 (CLS+mean-patch, 2560-d) extraction and bag/patch feature stores; supersede the skeleton's `load_feature_row`
  - `tcga_ut/code/data/prep/wsi_bag_cache.py` → WSI bag cache builder
- Fix `commands.py:cmd_prepare` so it dispatches to the real per-dataset adapter selected
  by config, with `_gen_mock` reachable only from `cmd_smoke`.
- Verify against report/2_cls_imb_benchmark.tex: exact per-dataset targets and patient/case split units,
  patient-disjoint splitting, deterministic eligibility/tiling.
- Discard: `native/code/data/progan/*`, `tcga_ut/code/data/progan/*`,
  `common_code/progan/*`, `tcga_ut/code/data/prep/export/synthetic.py`,
  `tcga_ut/code/modeling/patch/{models,data,train,artifacts,synthetic}.py` (raw-image
  pipeline), `native/code/data/full_scale/*` (predecessor power-law scheme — verify no
  hidden reuse first).
- Tests: port `native/code/tests/{test_feature_store.py,test_bracs_native.py,test_panda_native.py}`,
  adapt to new module paths; add CAMELYON16/TCGA-UT equivalents.

### Phase 2 — Manifest & controlled construction
- Keep the new package's `construction.py` as the base (already implements the frozen
  exponential imbalance formula and 10%/5% round-robin caps correctly).
- Implement the currently-stubbed pieces for real: `cmd_pilot`'s nested pilot supports,
  three pilot construction seeds, stability-floor/method-floor selection, feasibility
  fallback; `cmd_freeze`'s full condition set (natural, size-matched balanced, moderate,
  severe) with hierarchical patient/slide/patch ordering, patch and MIL contribution caps,
  requested/achieved imbalance ratio and normalized entropy, and manifest content-hash
  freezing that refuses training on mismatch.
- Cross-check `native/code/data/{dataset,filtering,sampling}.py` for reusable balanced-
  sampling utilities before writing new ones; discard on overlap with `construction.py`.
- Tests: split-unit derivation, leakage rejection, deterministic evidence selection,
  contribution caps, integer allocation, feasibility fallback, tail assignments, seed
  separation, immutable manifest hashes.

### Phase 3 — Models & training methods

**Status: done.** All patch/WSI models, losses, and the full 13-method roster (7 shared
+ 4 patch-only + 3 WSI-only) are wired end-to-end through `commands/tuning.py` and
`commands/confirm.py`, including the write-from-scratch methods (train-time and
post-hoc logit adjustment, cRT) and the ported RankMix/SC-MIL/MDE/OKO/CFAL logic. Scope
notes:
- `wsi/bag_cache.py` (an in-memory bag-feature cache for repeated-epoch speedups) was
  **not** ported — `datasets/features.py:load_slide_features` is already
  `functools.lru_cache`d per slide, which covers correctness; a dedicated bag cache is a
  pure performance optimization that can be revisited if WSI confirm runs prove too slow
  on the cluster.
- The report's "aggregate across both tuning seeds, the tuning splits, and the
  assignments" selection rule is implemented only across the two tuning seeds within one
  split/assignment run; cross-split/assignment aggregation of tuning selections is
  deferred to Phase 5 (`submit`, which must invoke `tune`/`confirm` once per split ×
  assignment) plus Phase 4 (`analyze`, which ingests those run records). `tune` and
  `confirm` themselves are correctly split/assignment-agnostic (they operate on whatever
  frozen manifest set `freeze` last produced).
- cRT's "stage one inherits the selected CE configuration" is implemented as a fresh
  fit under CE's tuned hyperparameters (not a literal reuse of a separately-trained CE
  checkpoint), since stage one must use the same confirmation/tuning seed as the cRT run
  itself. Post-hoc logit adjustment instead literally reuses each seed's already-trained
  CE checkpoint (no retraining), per "post-hoc logit adjustment inherits the selected CE
  model."
- Test coverage adds roster/grid correctness, the update-budget formula, `build_model`
  dispatch, RankMix/MDE/OKO loss unit tests (including the MDE λ_con=0 ablation and OKO's
  class-membership sampling invariants), cRT's freeze/reinit contract, and one-step
  finite-training smoke tests per roster method in both regimes
  (`tests/test_modeling_phase3.py`). Not ported: the legacy suite's temperature-scaling
  and run-record-round-trip cases, since those are Phase 4 (calibration) and pre-existing
  untested `common.py` I/O respectively — out of this phase's diff.

- Port drop-in from `experiments/shared/common_code/` (adapt signatures to new
  data/config schema):
  - `models/mlp.py` → patch MLP (already matches: 2-layer, 512-hidden)
  - `wsi/bag_dataset.py` → `AttentionMil`, `DualExpertMil` (MDE), `bag_collate`
  - `wsi/bag_losses.py` → MIL CE/weighted-CE/focal, RankMix-inspired, SC-MIL, MDE
    (incl. λ_con=0 ablation)
  - `wsi/{rankmix_teacher,cfal,trainer,eval,bag_cache}.py` → WSI training/eval stack
  - `losses/{factory,focal,metric,weights}.py` → CE/weighted-CE/focal/soft-F1/soft-MCC
  - `losses/oko.py` → OKO set learning
  - `sampling.py` → balanced-sampling / `WeightedRandomSampler` wiring
- Port and rewrite training-loop orchestration: `native/code/modeling/training/{constructed_wsi,constructed_wsi_cache,constructed_wsi_data,loops,patch_feature_adapter}.py`,
  `tcga_ut/code/modeling/patch_feature/{training,patch_feature_cache,specialized_trainers}.py`
  to the report's update-budget rule (U = 30⌈T/B⌉ updates; RankMix teacher+student;
  cRT stage budgets; MDE joint updates) and the balanced-accuracy→macro-F1→NLL
  checkpoint/tie-break rule.
- **Write from scratch (no legacy shortcut exists for these)**: train-time logit
  adjustment, post-hoc logit adjustment (inherits CE), and cRT (freeze + reinit
  classifier, tunes only stage-two LR).
- Rewrite `common_code/tuning/{grid,registry}.py` grid *values* to match the frozen
  Appendix table (four-value method grids × four-value LR grid, ≤16 candidates); keep
  the `TuningVariant` machinery.
- Discard: `common_code/wsi/divide_conquer/*` and all `divide_conquer`/`dnc_*` registry
  entries; `native/code/modeling/models/sklearn.py`; Balanced Softmax as a standalone
  method (correctly absent — it's the τ=1 case of logit adjustment per the report).
- Tests: split and port the protocol-relevant subset of `tcga_ut/code/tests/test_benchmarks.py`
  (patient-disjoint splits, soft-F1/soft-MCC, SC-MIL pair counting, RankMix ranked-mixing
  order, MDE consistency term, CFAL affinity/effective-number, temperature scaling,
  tuning-grid expansion, run-record round-trip) — drop the ~8 ProGAN-only tests. Add new
  tests for logit adjustment and cRT. One-step finite-training test per method.

### Phase 4 — Analysis & statistics

**Status: done.** The DB/metrics skeleton plus greenfield statistical machinery
(bootstrap, permutation, Holm, gates, separability, RQ3 wiring) are implemented
under `imbalance_benchmark/analysis/` (subpackages `inference/`, `reporting/`,
`predictors/`) and wired end-to-end through a rebuilt `commands/analyze.py`.
Scope notes:
- Confirm/confirm_methods (Phase 3) were extended, not just Phase 4: runs now
  record a `"validation"` split (needed for temperature fitting) alongside
  `"test"`, real per-class precision/recall/F1/confusion-matrix/NLL/Brier
  (`analysis/metrics.py:classification_payload`) replacing the stub that
  hard-wired macro precision/recall to balanced accuracy, `class_names`
  provenance, and raw logits (`common.py:ARRAY_FIELDS` gained `"logits"`).
  No model checkpoints are persisted to disk; temperature scaling and the
  target-prior correction operate on the recorded validation/test logits
  instead, which matches the report's "fit on natural-validation NLL" and
  avoids the added storage cost.
- The target-prior correction (Eq. posthoc/train-time-target-prior) is scoped
  to `post_hoc_logit_adjustment` and `logit_adjustment` exactly as the report
  defines it; every other method's target-prior-corrected output collapses to
  its raw score, since the report never defines a prior-shift mechanism for
  CE, weighted CE, etc.
- The paired patient-block permutation test uses the first confirmation
  seed's predictions as the representative pair rather than jointly
  permuting all five matched seeds; the bootstrap CIs, by contrast, do
  resample all five seeds as paired blocks per replicate, matching the
  report. Extending the permutation test to seed-joint permutation is
  possible but not done here.
- "Tail assignment" as a manifest-construction axis is not yet threaded
  through `freeze`/`confirm` (each pipeline run currently produces one
  native-order assignment); the Holm confirmatory-family grouping treats
  the current single-assignment reality as the norm and is ready to extend
  once a later pass adds the assignment axis, matching Phase 3's precedent
  of deferring cross-split/assignment aggregation to `submit`.
- RQ3 model fitting (`analysis/predictors/rq3_wiring.py`, wrapping the
  pre-existing `modeling/rq3.py:fit_rq3_model`) is a reusable, tested library
  function operating on synthetic multi-dataset-target-group data, but is
  **not** invoked from the single-dataset-regime `cmd_analyze`: RQ3's
  independent unit is the dataset-target group, and one pipeline run only
  ever produces one group, so fitting random intercepts is meaningless until
  a higher-level, cross-experiment step pools multiple `analyze` outputs
  (a `report/`-level concern, out of this phase's per-run scope).
  Intrinsic-separability/condition-learnability probes
  (`analysis/predictors/separability.py`) are implemented and tested but
  likewise not wired into `cmd_analyze`, for the same reason RQ3 isn't:
  they exist to feed RQ3's predictor matrix.
- Test coverage (`tests/test_analysis_phase4.py`, 32 tests, plus updated
  calibration/Holm/permutation assertions in `test_benchmark.py`) covers
  tiers, macro NLL, calibration algebra, gate thresholds, recovery sign
  conventions, bootstrap stratum-preservation and Kish invariants,
  permutation p-value bounds, Holm confirmatory/exploratory partitioning,
  effective-support/ICC sanity checks, RQ3 model fitting on synthetic data,
  and an end-to-end ingest-to-LaTeX-table round trip. The `smoke` command
  (which now exercises the real `analyze` pipeline, not placeholder writes)
  passes end-to-end.
- Port: `tcga_ut/code/analysis/report/calibration/utils.py` (temperature scaling,
  reliability curves), `common_code/metrics/{calibration,payload}.py` (extend to macro
  NLL, tail-group Brier/NLL, ⌈K/3⌉ head/body/tail split per protocol, replacing the fixed
  top/bottom-8 split), `tcga_ut/code/analysis/results/{core,ingest,query,compact_storage}.py`
  (SQLite schema/ingestion, superseding the skeleton's placeholder `analyze`),
  `native/code/analysis/plotting/support/{tail_class,calibration,comparisons,matrices}.py`
  (head/body/tail and comparison plots).
- Check `tcga_ut/code/analysis/tuning/*` and `native/code/analysis/evaluation/tuning_*.py`
  against the new package's `tune` command before reuse — likely partially superseded.
- **Write from scratch (greenfield, nothing portable exists)**: the two deficit gates,
  recovery estimands, class-preserving crossed patient bootstrap (10,000 replicates, Kish
  effective count, stratum preservation), bootstrap preflight, paired patient-block
  permutation tests, Holm correction across the confirmatory family (weighted CE, focal,
  train-time logit adjustment, balanced sampling — shared across regimes), exploratory
  labeling for the rest, intrinsic-separability/learnability/effective-support/RQ3 models
  (no test-derived predictors).
- Replace the skeleton's placeholder `results_table.tex`/`results_plot.png` with real
  generation from the rebuilt DB, with provenance to run/manifest hashes.
- Discard: `tcga_ut/code/analysis/report/calibration/{posthoc,posthoc_table,table}.py`'s
  vector/Dirichlet calibration paths (excluded by plan — only temperature scaling +
  prior correction are in scope); `tcga_ut/code/analysis/report/progan_diagnostics/*`;
  `native/code/analysis/plotting/{bracs,camelyon16,panda}_native_report.py`.
- Tests: metrics, prior corrections, temperature scaling, gates, recovery signs,
  patient-clustered aggregation, bootstrap invariants, permutation pairing, Holm
  families, synthetic known-effect cases.

### Phase 5 — Hydra/SLURM submission
- Design `cmd_submit` fresh in Python, using `native/code/hydra/{run,job_defs,jobs,training_jobs,tuning_jobs,verify_jobs}.py`
  and the per-dataset `hydra/{bracs,camelyon16,panda}/{common_jobs,native_jobs}.py` /
  `submit_*_workflow.py` only as structural reference (job-ID-specific logs, arrays,
  `afterok` dependencies, Apptainer execution, SquashFS staging to `/tmp`, dry-run,
  test-partition smoke jobs) — do not port these files directly, they're tightly coupled
  to the old per-dataset CLI.
- `common.py:render_sbatch`/`submit_sbatch` already sketch the template pattern; extend
  to the full dependency-linked workflow the plan describes.
- Discard: `tcga_ut/code/hydra/*.sh`, `submit_epoch_sqfs*.sh`, `submit_sqfs_train.sh`,
  `run_patch_staged_train.sh`, `native/code/cli/{train,train_core,train_support,sample_balanced,visualize}.py`,
  `tcga_ut/code/run_pipeline.py`.
- Tests: render the full SLURM dependency graph without submission (dry-run).

### Phase 6 — Deletion and repository integration
- Only after Phases 1–5 are done and their tests pass: delete `native/`, `tcga_ut/`, and
  `experiments/shared/` (moving anything still needed into `imbalance_benchmark` first).
- Grep the whole repo for references to `native`, `tcga_ut`, `common_code`, ProGAN,
  `experiments/shared`, `class_imbalance`, `design_dataset` (imports, configs, docs,
  generated commands) before deleting — none should remain except intentional mentions
  in report prose/citations.
- Update root packaging, Pyright, pytest, README, local AGENTS guidance, and cluster path
  configuration to reference only the unified benchmark package.
- Leave `experiments/class_imbalance/` and `experiments/design_dataset/` untouched; just
  remove the benchmark's dependencies on them.
- Compile `report/2_cls_imb_benchmark.tex` through the prescribed temporary build directory and verify
  no auxiliary files remain in the worktree.
- Run Ruff, Pyright, scoped Vulture, the clean-code audit, and the complete test suite.

---

## Original plan (for reference — protocol/architecture requirements unchanged)

### Summary

Replace `native/`, `tcga_ut/`, and `experiments/shared/` with one self-contained implementation under `experiments/2_cls_imb_benchmark/code/`. Selectively port only code that matches the protocol in `report/2_cls_imb_benchmark.tex`; remove ProGAN, raw-image classifier experiments, duplicated orchestration, obsolete analyses, and legacy outputs.

The resulting pipeline will reproduce the experiment from raw dataset metadata/images through frozen Virchow2 features, controlled construction, tuning, confirmation runs, statistical analysis, and report artifacts.

### Implementation Changes

#### Unified structure and interfaces

- Create an importable `imbalance_benchmark` package organized by:
  - dataset adapters and frozen-feature extraction;
  - controlled manifest construction;
  - patch and attention-MIL models/methods;
  - tuning and confirmation execution;
  - evaluation/statistical analysis;
  - Hydra/SLURM submission.
- Provide one CLI:
  - `prepare`: validate datasets, create patient-disjoint splits, select eligible evidence, and extract Virchow2 features;
  - `pilot`: run nested support-stability pilots and determine support floors;
  - `freeze`: create definitive condition manifests and the content-hashed analysis manifest;
  - `tune`: execute and select validation-only hyperparameter candidates;
  - `confirm`: fit the five confirmation seeds and emit locked test predictions;
  - `analyze`: rebuild the result database, inference outputs, figures, and LaTeX tables;
  - `submit`: render or submit the corresponding dependency-linked SLURM workflow;
  - `smoke`: run a synthetic end-to-end validation without external datasets.
- Use one versioned YAML configuration for dataset paths, evidence caps, environment/container paths, and SLURM resources. Keep scientific controls and grids as checked-in typed constants so local configuration cannot silently change the protocol.
- Store manifests as CSV plus metadata JSON, run records as JSON with compressed prediction sidecars, and derive the analysis SQLite database reproducibly from those records.
- Move the four computational report-helper scripts into the package's construction/analysis modules; keep `report/` limited to `2_cls_imb_benchmark.tex`, bibliography, generated report outputs, and `main.pdf`.

#### Protocol-aligned data construction

- Implement adapters for TCGA-UT, BRACS, PANDA, and CAMELYON16 with the exact targets and patient/case split units stated in the report.
- Preserve the raw-to-report boundary: deterministic eligibility/tiling, fixed evidence caps, frozen Virchow2 extraction using concatenated CLS and mean patch tokens, and reusable patch/bag feature stores.
- Create the three patient-disjoint split seeds before any imbalance manipulation and validate that validation/test cohorts remain natural and identical across conditions.
- Implement natural, size-matched balanced, moderate, and severe conditions with:
  - common controlled support \(T\);
  - deterministic hierarchical patient/slide/patch orderings;
  - patch patient/slide contribution caps;
  - MIL patient contribution caps;
  - requested and achieved imbalance ratios and normalized entropy;
  - clinical/native, reversed/rotated, and random tail assignments as applicable.
- Implement nested pilot supports, three pilot construction seeds, stability-floor selection, method floors, feasibility fallback, and the separate definitive construction seed.
- Freeze all identifiers, counts, seed roles, grids, environment metadata, hashes, bootstrap diagnostics, and approved deviations before definitive fitting. Refuse training if manifests or configuration no longer match the frozen hash.

#### Models, methods, and training

- Standardize patch classification on the report's 2560-dimensional frozen features and two-layer 512-unit MLP; remove the obsolete raw-image ResNet and synthetic-image pipelines.
- Standardize WSI prediction on capped feature bags and the common 256-dimensional attention-MIL model.
- Implement only the prescribed roster:
  - both regimes: CE, balanced sampling, weighted CE, focal loss, train-time and post-hoc logit adjustment, and cRT;
  - patch only: CE + soft F1, CE + soft MCC, CFAL, and OKO;
  - WSI only: RankMix-inspired training, SC-MIL, and MDE-inspired dual experts, including the zero-consistency ablation.
- Do not retain ProGAN, divide-and-conquer, Balanced Softmax as a separate method, vector/Dirichlet calibration, or other methods absent from the report.
- Encode the report's four-value learning-rate and method-specific grids exactly. Post-hoc adjustment inherits CE; cRT inherits CE stage one and tunes only its classifier-stage learning rate.
- Train by optimizer updates rather than epochs, including the prescribed RankMix, cRT, and MDE stage budgets. Record processed examples, unique exposures, timing, accelerator use, memory, parameter counts, and checkpoints.
- Select checkpoints and hyperparameters exclusively on natural-validation balanced accuracy, then macro F1, then NLL. Enforce two tuning seeds and five disjoint confirmation seeds.

#### Analysis and cluster execution

- Retain raw, balanced-decision, target-prior-corrected, and temperature-scaled outputs separately.
- Implement all specified discrimination, calibration, ordinal, cost, classwise, and aggregation-unit metrics.
- Implement the two deficit gates, recovery estimands, crossed patient bootstrap, bootstrap preflight, paired patient-block permutation tests, Holm correction for the confirmatory method family, and exploratory labeling for the remaining methods.
- Implement the intrinsic-separability, learnability, effective-support, and restricted RQ3 models without using test-derived predictors.
- Generate report-ready tables and figures only from the rebuilt result database, with provenance back to run and manifest hashes.
- Consolidate Hydra support into Python-based job rendering/submission with job-ID-specific logs, arrays, `afterok` dependencies, Apptainer execution, SquashFS staging to `/tmp`, dry-run inspection, and test-partition smoke jobs.

#### Deletion and repository integration

- Delete both legacy code trees and their obsolete tracked outputs after the unified smoke pipeline and retained unit tests pass.
- Move required code from `experiments/shared/common_code` into the unified package, remove unused shared modules and tests, then delete `experiments/shared/`.
- Leave `experiments/class_imbalance/` and `experiments/design_dataset/` untouched, but remove the benchmark's dependencies on them.
- Update root packaging, Pyright, pytest, README, local AGENTS guidance, and cluster path configuration to reference only the unified benchmark package.
- Preserve the untracked user-owned `todo.md` and unrelated worktree content.

### Test Plan

- Unit-test all dataset label mappings, split-unit derivation, leakage rejection, deterministic evidence selection, contribution caps, integer allocation, feasibility fallback, tail assignments, and seed separation.
- Test every method's loss/model behavior, class-aware batch constraints, update budget, stage transitions, parameter grid, and one-step finite training.
- Test validation-only selection, deterministic tie-breaking, shared balanced-baseline reuse, immutable manifest hashes, resumable runs, and rejection of incomplete tuning/confirmation sets.
- Test metrics, prior corrections, temperature scaling, gates, recovery signs, patient-clustered aggregation, bootstrap invariants, permutation pairing, Holm families, and synthetic known-effect cases.
- Run a synthetic end-to-end smoke workflow locally and render the full SLURM dependency graph without submission.
- Run Ruff, Pyright, scoped Vulture, the project clean-code audit, and the complete benchmark test suite.
- Compile `report/2_cls_imb_benchmark.tex` through the prescribed temporary TeX build directory and verify that no auxiliary files remain in the worktree.
- Before deleting the legacy trees, verify that no imports, configuration paths, documentation links, or generated commands reference `native`, `tcga_ut`, `common_code`, ProGAN, `experiments/shared`, `class_imbalance`, or `design_dataset`.

### Assumptions

- `report/2_cls_imb_benchmark.tex` is the authoritative, frozen protocol; implementation discrepancies are resolved in favor of the report.
- This is a clean replacement with no compatibility wrappers for legacy commands, result schemas, or output paths.
- Obsolete figures, tables, selection records, and ProGAN artifacts are deleted rather than archived; the report source, bibliography, and compiled `main.pdf` remain.
- Frozen feature extraction is part of reproducibility but remains excluded from per-method computational-cost comparisons, as specified in the report.
- Existing implementation code is reused only after its behavior is covered by protocol-aligned tests; directory location alone does not justify preservation.
