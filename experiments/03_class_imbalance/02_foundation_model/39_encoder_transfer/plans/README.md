# Experiment 39: does imbalance damage transfer across foundation models?

Status: prospective plans only. No model download, feature extraction, training, or cluster verification has been performed for this experiment. Repository evidence and the UNI2-h model card were inspected on 2026-09-24.

**Question:** On identical patients and image patches, does replacing frozen Virchow2 with frozen UNI2-h change the loss of balanced accuracy caused by class imbalance, and does that change arise in the prior-only channel, the support-only channel, or their interaction?

This is more informative than a leaderboard comparison. A stronger balanced encoder can still lose the same amount under imbalance. Conversely, a smaller damage estimate can result from a worse balanced baseline. The experiment measures both absolute accuracy and the within-encoder damage, then compares those damages between encoders.

The primary endpoint is the encoder difference in combined damage at ratio 100, separately for BRACS and TCGA-UT. Ratio 10 supplies a moderate-imbalance check. The prior/support decomposition explains which controlled channel changes; calibration and representation diagnostics remain secondary. This tests transfer of the existing conclusions to one alternative encoder, not foundation models in general.

## Work packages

Execute in order. Each file defines concrete outputs and an exit gate. Check off work only when evidence exists.

1. [Freeze the scientific protocol](01_protocol.md): estimands, arms, inference, and claim boundaries.
2. [Audit inputs and obtain the encoder](02_inputs_and_model.md): immutable patch identities, authenticated access, model/environment lock, resource pilot.
3. [Extract and validate features](03_feature_extraction.md): separate UNI2-h cache, exact image correspondence, resumable extraction.
4. [Implement and verify the experiment](04_implementation.md): reuse existing allocator/readout, pair encoders, preserve provenance, focused tests.
5. [Execute on Hydra](05_execution.md): smoke tests, resource estimates, bounded submissions, complete evidence.
6. [Analyze and write the report](06_analysis_and_report.md): paired contrasts, qualified interpretation, generated tables, compiled PDF.

Future implementation lives beside `plans/` in `code/`, `configs/`, `tests/`, and `report/`; large outputs remain on Hydra. Commands in these plans that refer to experiment 39 are proposed interfaces, not currently runnable commands.

## Why this experiment now?

Repository sources, relative to `experiments/`:

- `03_class_imbalance/00_damage/25_damage_tcga_ut/report/results.tex` and `26_damage_bracs/report/results.tex`: at ratio 100, observed damage was 2.91 pp on TCGA-UT and 7.65 pp on BRACS. BRACS was already damaged at ratio 2. Validation temperature scaling removed most of the measured calibration increase.
- `03_class_imbalance/01_cause/tcga_ut/27_prior_vs_support/report/27_prior_vs_support.tex` and `bracs/28_prior_vs_support/report/28_prior_vs_support.tex`: TCGA-UT's 2.91 pp decomposed into prior 1.20, support 0.53, and interaction 1.18 pp. BRACS's 7.65 pp decomposed into 4.47, 3.29, and -0.11 pp; its interaction was unresolved, not established as zero.
- `03_class_imbalance/01_cause/cross/36_joint_mechanism/report/36_joint_mechanism.tex`: apparent mechanism rescue depended on regularization; the pilot failed integrity, robustness, and precision gates. Do not equate an accuracy improvement with reduced imbalance damage.
- `03_class_imbalance/01_cause/cross/37_support_coverage/report/37_support_coverage.tex`: the coverage intervention failed its precheck. Coverage remains a hypothesis.
- `03_class_imbalance/01_cause/cross/38_lambda_selection/report/38_lambda_selection.tex`: a test-set oracle did not establish that better lambda selection closes the prior gap. Its uncertain pilot result does not prove an encoder-independent intrinsic mechanism.
- `01_benchmark/AGENTS.md` and `02_patient_shortage/AGENTS.md`: earlier work distinguishes nominal weighting effects from lost patient support. Breadth and centre estimation matter under frozen Virchow2. Here patient breadth stays fixed within each dataset, so changing it cannot explain an encoder contrast.

No native-prevalence arm, tail-order search, synthetic separation intervention, fine-tuning, additional classifier, or third encoder is needed for this question. Those would expand the interpretation and compute burden without identifying the primary contrast more clearly.

## Completion checklist

- [ ] Protocol and immutable inputs frozen before main outcomes.
- [ ] UNI2-h access, exact checkpoint, transforms, and container verified.
- [ ] Both datasets have complete, identity-matched feature caches.
- [ ] Focused tests and scoped clean-code checks pass.
- [ ] All planned main cells complete and pass provenance/convergence checks.
- [ ] Analysis and report produced from measured records; PDF visually checked.

External source: [official UNI2-h model card](https://huggingface.co/MahmoodLab/UNI2-h). Access and extraction details are in phase 02; no token is assumed verified.
