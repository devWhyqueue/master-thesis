# Phase 04 — implement paired fits and analysis

Input: frozen design and accepted feature caches. Output: one small experiment package, two dataset configs, focused tests, and a working dry run. Python implementation is future work.

## Reuse boundaries

Reuse the allocator and weighting logic in `03_class_imbalance/00_damage/25_damage_tcga_ut/code/prevalence/fit.py`: `class_permutation`, `class_counts`, `_shard_context`, and `_prior_weights`. Reuse arm semantics from `01_cause/tcga_ut/27_prior_vs_support/code/cause/fit.py`. Both are relative to `experiments/`.

Do not invoke their complete runners unchanged: they enumerate older arm sets, fixed draw counts, and old output paths. Existing cause code also borrows stored ratio-arm predictions. Experiment 39 must fit B/P/S/R consistently for each encoder and grid; old reported scores are historical context only.

Reuse `02_patient_shortage/00_damage/05_effective_support/code/breadth/fit.py` and `breadth/calibrate.py` for tuning/calibration, and `01_cause/03_classifier_limitation/code/decodability/linear.py` for the probe. Existing `init_shard` resolves benchmark paths and `tune_and_fit_draw` uses module-level lambda constants; expose only the explicit inputs experiment 39 needs or use a thin local runner around the lower-level solver. No global monkeypatching of dimensions, grids, or encoder configuration.

Audit `breadth/sampling.py`, `ImbalanceDataset`, freeze metadata, and all `INPUT_DIM` assumptions along the real load/evaluation path. Raw tensor loading can already accept differing widths, but provenance and higher-level contracts still require checks. Never infer encoder identity from vector width alone.

Reuse patient-macro recall, probability metrics, atomic run records, and bootstrap utilities. Check multi-label patient handling and repeated patients across splits before reusing bootstrap output directly. Exp-25's same-encoder paired analysis does not automatically implement an encoder difference-in-differences.

## Proposed interface and files

Use the repository's `python code --config configs/<dataset>.yaml <stage>` convention. Implement stages `preflight`, `extract --shard-index`, `audit-features`, `fit --shard-index`, `analyze`, and `submit --stage ... --dry-run`. These are specifications, not existing CLI commands.

Keep `code/__main__.py`, a small `encoder_transfer` package, and `configs/{bracs,tcga_ut}.yaml`. Separate extraction from fitting only where their dependencies differ. Per fit shard, address one dataset/encoder/split/draw and fit the seven arms. Include every selected candidate and B-lambda sensitivity without launching redundant fits.

Each run record must include: protocol/config/source-code hashes; encoder/cache/input locks; class order; patient/patch IDs or immutable manifest references; split/draw; arm; realized counts and row weights; every lambda's convergence and validation score; coefficients/intercepts; selected lambda; validation temperature; ordered test identities, labels, predictions, and probabilities. Failed candidates and failed arms remain visible.

Keep temperature-scaled probability storage compatible with existing helpers. Bound candidate storage by retaining coefficients and validation scores, not full test probabilities for every lambda. Generate sensitivity predictions from coefficients when needed. Prevent reuse across a changed config, model, image manifest, or code version.

## Focused checks

Use small synthetic data and existing test patterns. Test behavior, not a duplicate copy of the implementation:

- B/P use identical IDs; S/R use identical IDs; every paired encoder arm has identical IDs and weights. Verify positive class support and fixed total budget.
- P weights reproduce requested shares; S weights reproduce uniform shares; both sum to N. Equal-ratio input makes all four arms coincide.
- Swapping features for a copy of the same encoder produces exactly zero encoder damage contrast; the decomposition identity holds. Known toy predictions verify sign and percentage-point units.
- Paired patient weights are shared across encoders/arms and across split occurrences; multi-class BRACS patients remain one cluster. Row-order changes cannot change metrics after ID alignment.
- 1,536- and 2,560-dimensional matrices run through the same probe; wrong provenance is rejected. Test ties, nonconvergence, incomplete candidate grids, and stale-result refusal.
- Temperature is selected without test labels, and positive temperature leaves predictions unchanged. Check existing NLL/ECE weighting and record the definitions in the protocol lock.
- Simulate interrupted extraction/resume and one tiny end-to-end fit/analyze path; confirm incomplete evidence prevents a completed report.

After Python edits, run affected tests and the clean-code skill with appropriate `--scope` and `--vulture-scope`. Read its rules before editing. Do not run a separate full pytest suite. Exit gate: focused checks, scoped clean-code checks, and submission dry-run pass; existing benchmark defaults and frozen outputs remain intact.
