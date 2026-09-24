# Phase 06 — analysis and concise report

Input: accepted main records. Outputs: machine-readable `analysis.json`, generated tables/figures, editable LaTeX, bibliography, and one compiled, visually checked PDF. All findings below are questions to evaluate, not predicted outcomes.

## Analysis sequence

1. Recheck input/candidate completeness and exact pairing. Record balanced and imbalanced accuracy before interpreting damage differences.
2. Compute primary delta_R at 100 for each dataset with the phase 01 paired bootstrap and multiplicity rule. Report point estimates, uncertainty, the 1 pp practical threshold, and per-split effects.
3. Decompose delta_R into delta_P, delta_S, and delta_I. Compare tuned and encoder-specific B-lambda sensitivities. Avoid announcing a prior/support mechanism that reverses with this choice.
4. Show ratio 10 as a secondary severity check. Two nontrivial ratios cannot identify a smooth onset curve or establish acceleration.
5. Report raw/scaled macro NLL and ECE for B and R at both ratios, with paired encoder differences and differences in damage. Retain natural metric units; no arbitrary shared practical threshold with BA. Balanced NLL and ECE answer related but distinct questions.
6. Examine per-class recall and the fixed tail third from the frozen allocation permutation. A lower damage score with a worse B arm is not evidence of better usable robustness; absolute R performance resolves that ambiguity.

## Limited explanation of why

The P/S interventions support attribution to the operational loss-prior channel, support redistribution, or their nonadditivity. They do not uniquely identify a biological or representation mechanism.

Without extra training fits, describe feature norms, validation balanced recall, standardized validation logit margin, and classwise confusion for each encoder. Define margin as true-class logit minus strongest competing logit, divided by that model's validation margin standard deviation; report raw values too. Handle zero variance explicitly. Use validation diagnostics to formulate explanations, never to select the favorable test contrast. Comparisons across encoders remain exploratory because geometry, dimension, training data, and pooling all change together.

Do not repeat exp-33's unvalidated injected-prior proxy as measured prior damage. Do not reuse test-derived worst-tail orders, label-aware evaluation feature interventions, or richer reference centres as if they were part of a deployable encoder comparison. A separate intervention would be needed to establish geometry as the causal mediator.

## Report shape

Target a short article, roughly six pages plus compact references; clarity takes precedence over an exact page count. Use scientific-writing skill for prose. Keep `report/39_encoder_transfer.tex` as the sole build root, `results.tex` for measured findings, and generated assets in `report/`.

The introduction states the transfer question and gives the established Virchow2 damage/decomposition numbers from experiments 25–28. Briefly connect earlier patient-shortage work: broad independent support matters, but this design fixes patient identities. Mention regularization sensitivity and failed coverage gates as limitations of existing explanations, not as confirmed mechanisms.

Methods state the paired image design, different within-dataset support budgets, model/transform provenance, four-arm decomposition, validation-only selection/calibration, and conditional bootstrap inference. Distinguish fresh training draws from reused test patients.

Use three principal displays:

- An accuracy/damage figure: B, R10, and R100 per encoder and dataset, plus the two primary damage differences with intervals.
- A decomposition figure or compact table: D_P, D_S, I, and D_R at 100, with encoder differences and a small fixed-B sensitivity panel.
- A probability-quality table: raw and temperature-scaled NLL/ECE at B and R100; ratio 10 and class diagnostics can be appendix material.

Discussion answers three separate questions: Did balanced performance change? Did imbalance damage change? Which operational channel changed? If the primary interval is unresolved, say so before discussing patterns. Practical similarity requires the predeclared interval rule; failure to reject zero is insufficient.

State limitations: only two encoders, one linear readout, fixed patients/splits, dataset-specific support budgets, different native embeddings/transforms, possible undocumented pretraining overlap, and no causal isolation of representation geometry. Do not generalize results to foundation models as a category.

## Verification and handoff

Generate every result table from `analysis.json`; never type plausible numbers into result placeholders. Before execution, any prose must explicitly say prospective. After execution, trace each headline to an estimate and interval, including null/unresolved outcomes and protocol deviations.

Add verified citations for UNI2-h's exact checkpoint/model card, Virchow2, dataset sources, and relevant local experiment reports. Distinguish the UNI2-h checkpoint documentation from the earlier UNI paper linked by its card.

Compile through Git Bash using `scripts/tex-build.sh experiments/03_class_imbalance/02_foundation_model/39_encoder_transfer/report/39_encoder_transfer.tex`. Inspect build diagnostics, render PDF pages, and visually check tables, labels, legends, clipping, and readable intervals. Run `git diff --check` and verify all generated numbers against analysis outputs.

Exit gate: editable sources, auditable measured outputs, and visually checked PDF delivered together. No report claim is marked established merely because its planned experiment completed.
