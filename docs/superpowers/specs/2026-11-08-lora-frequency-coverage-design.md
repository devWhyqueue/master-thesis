# Four-condition LoRA decomposition

Status: accepted as the basis for implementation planning on 2026-11-08; no implementation or cluster submission.

## Intent

Repeat the frequency/coverage decomposition with the learner used for mitigation. The presentation should connect measured causes to the recovery that rebalancing might achieve without transferring numerical shares from frozen logistic regression to LoRA.

The user chose the four-condition experiment after considering reuse of existing mitigation runs and requested the implementation plan after reviewing this design. Implementation and execution remain the next phase.

## Scope

- Proposed experiment: `experiments/03_class_imbalance/01_cause/cross/48_lora_decomposition`.
- BRACS and TCGA-UT, Virchow2, imbalance ratio rho=100 only.
- Reuse the existing three patient splits and ten allocation draws per split: 30 matched four-arm blocks per dataset, 240 fits total.
- Preserve the mitigation allocations: 10 training patients/class and 2,240 patches on BRACS; 20 patients/class and 19,200 patches on TCGA-UT. Reconstruct the same patient selections, patch ordering, and class-to-rank permutations through the existing allocation helpers.
- Same held-out validation/test patients, class order, preprocessing, pretrained checkpoint, and patient-macro balanced accuracy as the mitigation studies.
- No additional encoders, imbalance ratios, method grids, representation interventions, or search for a more favorable recovery ceiling.

## Approaches considered

1. Reuse balanced CE, imbalanced CE, and fixed-strength balanced sampling (s=1). Cheapest way to obtain a LoRA rebalancing benchmark; lacks the frequency-only condition and cannot separate frequency from interaction.
2. Complete a four-arm study, reusing those three runs and adding frequency-only training. Saves fits, but balanced sampling changes patch exposure and the existing artifacts do not establish matched initialization across all four arms.
3. Fresh matched four-arm study using loss weighting, following the original decomposition. More expensive, but controls patch exposure within each allocation and initialization across conditions. This is the proposed implementation of the user's selected four-condition approach.

## Conditions and control

Each arm trains its own LoRA adapters and linear head from the same pretrained encoder; none continues from another arm's fitted model.

| Arm | Available patch allocation | Target class share in loss |
| --- | --- | --- |
| B: balanced reference | balanced r1 | uniform |
| F: frequency only | balanced r1 | realized r100 shares |
| C: coverage only | imbalanced r100 | uniform |
| R: both | imbalanced r100 | realized r100 shares |

All arms sample uniformly with replacement from their own available rows. For class c with n_c rows among N, and target class share q_c, use the fixed row loss weight w_c = q_c / (n_c/N). Compute mean(w_y * per-example CE) over the batch, rather than dividing by the random sum of weights in that batch. The dataset-average weight is one; B and R reduce to ordinary CE. Target r100 shares come from the paired realized allocation, not an idealized exponential profile.

Use the same sampled row-index sequence for B/F and for C/R. Thus changing frequency does not also change the distinct patches actually encountered within an allocation. Coverage here concerns the available patch allocation, not necessarily every row being visited. Record realized unique exposure as a check.

Explicitly seed adapter/head initialization and all random sources using a recorded seed derived from (split, draw), shared by the four arms. Reset initialization before each arm. A sampler seed alone is insufficient. Different blocks use different seeds. Pairing reduces avoidable training variation but does not promise narrower patient uncertainty.

## Training recipe and execution

Reuse exp-40 training, image loading, encoder, evaluation, and record-writing machinery with the smallest explicit extension needed for fixed loss weights. Existing mitigation methods retain their current behavior. No new training framework or dependency.

Use the exp-41/43 recipe: LoRA rank 8, alpha 16, dropout 0; linear head; batch 256; AdamW learning rate 1e-4 and weight decay 1e-4; existing cosine schedule. BRACS uses 600 updates, TCGA-UT 2,400. Use the existing deterministic image preprocessing; no CUDA augmentation, balanced sampling, or method-specific loss. These are fixed-recipe effects, not claims about the best achievable model for each condition.

Start with draw 0 of each split on both datasets: 24 fits. Check allocations, class weights, identical initialization and within-allocation index streams, finite losses/outputs, prediction identities, and artifact completeness. Inspect training/validation behavior for numerical or optimization failure; do not gate continuation on the sign, size, or significance of the scientific effect. If the implementation and recipe are sound, retain these blocks and complete the remaining 216 fits. If a recipe change is required, document it and restart all affected conditions consistently before pooling results.

Use fresh experiment outputs and leave mitigation artifacts unchanged. Historical B/R results are descriptive reproducibility references; new estimates use only the fresh paired blocks. Execution must follow the project hydra-cluster skill and current account-wide queue limit. No submission is authorized by this design document alone.

## Estimands

Let B, F, C, R denote pooled patient-macro balanced accuracies in percentage points.

- Total damage: D = B - R.
- Frequency-only damage: D_F = B - F.
- Coverage-only damage: D_C = B - C.
- Interaction: I = D - D_F - D_C.
- Recovery from restoring balanced class influence: G = C - R = D_F + I.
- Corresponding share: h = G / D = 1 - D_C / D.

Absolute damage and recovery are primary results; h is the presentation-facing secondary result. Report h with paired uncertainty, without truncating it to [0,1]. If D's 95% interval reaches zero or its point estimate is below the project's 1 pp materiality threshold, omit a numerical ceiling claim and report the absolute contrasts instead. Do not silently discard nonpositive-denominator replicates or replace the random denominator by a constant.

Reuse the existing paired patient/bootstrap and draw-pooling conventions, sharing patient weights and resampled draw indices across all four arms. Retain the existing handling of repeated patients across overlapping splits; audit it before relying on it. Preserve patient-level dependence rather than treating patches or overlapping splits as independent observations. Report 95% intervals, per-split estimates, and the conditional scope of inference. Separate patient-only and draw-only resampling summaries as uncertainty diagnostics, not substitutes for the full intervals or additive variance components. One training seed per allocation draw means draw variation includes initialization variation; do not label it pure allocation variation.

## Interpretation

The result is a matched LoRA decomposition and a benchmark for restoring balanced class influence under this fixed training recipe. It is not an irreducible information limit or an upper bound on augmentation, different losses, or better optimization. It can support a conditional hypothesis: if a method removes frequency-related damage while leaving this coverage residual unchanged, its recovery should be near h.

This follow-up is motivated by already-observed mitigation results. It cannot retrospectively turn the benchmark into a prospective prediction of those results. Compare against mitigation descriptively, and distinguish plain recovery from recovery corrected for a method's balanced-data gain.

## Deliverables and validation

- Scoped experiment code/configs, provenance for each arm, and paired analysis JSON.
- A concise scientific protocol/results report with arm accuracies, damage components, interaction, recovery benchmark, and uncertainty diagnostics. Compile and visually inspect its PDF using repository conventions.
- Tests of allocation/weight identities, equal-loss reduction for B/R, initialization and sampling pairing, weighted-loss normalization, and contrast algebra including an unstable denominator. Run affected tests and the mandatory clean-code skill after Python changes.
- Do not alter the existing summary's numerical claims until the experiment has completed and the results have been reviewed. No automatic push.

The builder may choose names, CLI wiring, and the smallest compatible shared helper extension. Any change to datasets, encoder, rho, training budget, arm definitions, or uncertainty target requires an explicit design revision.
