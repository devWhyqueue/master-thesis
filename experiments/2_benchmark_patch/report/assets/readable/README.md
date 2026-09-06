# Readable report appendix

Run `node assets/build_readable_appendix.mjs` from the report directory to regenerate the numerical fragments and the two TikZ figures. The builder uses only the vendored `assets/tables/*.tex` files, retains the original fragments, and checks source counts and the TCGA-UT moderate estimates. It does not run models or recompute inference.

- Damage and partition tables convert balanced accuracy from fractions to percentage points. Calibration damage remains in nats.
- Recovery matrices convert fractions and paired intervals to percentages and retain Holm-adjusted permutation p-values. Only source-eligible comparisons appear. The overview marks adjusted p-values below 0.05; colour saturates at ±100%, but annotated estimates are not truncated.
- Allocation and tuning tables give observed minimum–maximum ranges. Tuning counts include repeated assignment records, not independent searches.
- Roster ranges span eligible assignment-level discrimination estimates. They are not confidence intervals. The original exploratory NLL and interval records remain in `assets/tables/roster_recovery.tex`.
- Tier tables pivot all 72 reported tier-recall rows without pooling class–tier records. Full class, F1, NLL, and Brier records remain in the original fragments.
- Calibration tables retain CE aggregate raw/scaled NLL and ECE. The scientific interpretation in `calibration_interpretation.tex` is maintained prose, extracted from the results report; the builder does not overwrite it.
- Cost ranges span matched point differences across recorded conditions and assignments, converting hours to minutes and bytes to MiB. They are not confidence intervals; the original intervals remain in `assets/tables/cost.tex`. Checkpoint-reuse costs are incremental, not total training costs.

`natural_anchor_summary.tex` uses the per-partition patch totals from `realized_support.tex`; these are allocation sizes, distinct from exposure budgets. Figures use the same rounded source estimates as the report tables, with no additional statistical claims.
