# Design-Dataset Report — Future Directions

Notes on what the current report (`main.tex`) does *not* yet cover, relative to
the original draft (`Dataset TCGA_UT.pdf`), plus a clean methodology for the
class-difficulty axis if/when we pick it back up.

## Dropped from the original draft

The new report took the "controlled severity sweep + many methods + calibration"
direction and dropped several distinctive ideas from the original:

1. **Two-axes framing: class difficulty vs. frequency.** The conceptual backbone
   of the original (Sec. 3.1) is gone; imbalance is now treated purely as a
   frequency knob.
2. **Inherent class-difficulty estimation** via simple classifiers (KNN/NCC/MLP)
   on balanced features — the tool used to *measure* difficulty.
3. **Easy-to-Difficult / Difficult-to-Easy class orderings.** New report uses
   only native-prevalence order. The original's most novel experiment
   (deliberately oversampling difficult vs. easy classes to disentangle
   difficulty from frequency; finding that "difficult-to-easy" is the desirable
   real-world target) is absent.
4. **Per-class recall analysis + scatter plots** (recall at λ vs. λ=0). New
   report reports only aggregate metrics, losing the per-class collapse story.
5. **Confusion matrices** (original Sec. 3.1.2) — not present.
6. **λ=0 balanced anchor.** Mild end is now λ=0.5; reference is "native
   full-pool." No clean fully-balanced baseline.
7. **Foundation-model robustness / feature separability of rare classes**
   (investigation goal (a)) — no explicit feature-geometry analysis.

Note (not a gap): the original's iterative **KL-projection onto the closest
feasible distribution** was replaced by shrinking N (feasibility search) so no
target overflows. Cleaner design choice; worth one sentence if a reviewer asks
why not project.

## Clean methodology for the class-difficulty axis

Problem with the original approach: it conflated "measure difficulty" with
"build a balanced dataset." Fully balancing TCGA-UT drops to the tail size
(~20 slides/class) → few-shot + ~98% of data wasted.

**Reframe:** difficulty is a property of the frozen Virchow2 feature geometry per
class, not of the trained classifier. Balance the *probe budget*, not the
*dataset*. Measure difficulty once on a tiny fixed probe; leave the rest of the
data for the imbalance experiment.

### Primary: nearest-centroid (prototype) recall

A trained MLP's boundary is biased by full class counts → frequency-confounded.
A nearest-class-mean classifier is symmetric across classes by construction; its
only frequency sensitivity is prototype-estimation variance, which we equalize.

- Estimate each class prototype from the **same `k` slides** (`k` ≈ tail size,
  ~15–20). Equal budget → no estimation-variance confound.
- Evaluate per-class recall on **all remaining slides of every class** — head
  classes keep contributing, so nothing is wasted on evaluation. Only `k·C`
  slides are "spent" on prototypes.
- **Bootstrap** the `k`-slide draw (resample prototypes many times) → per-class
  difficulty with CIs, and uses far more head data across draws.

Difficulty = mean held-out recall per class. Frequency-invariant by
construction, cheap because features are frozen.

**Caveats:**
- **Pseudo-replication.** Patches from one slide are not independent. Work at
  slide level (or slide-disjoint splits) or a 20-slide tail class looks like
  thousands of points. It has ~20 — tail-class difficulty CIs will be wide.
  Report them; that width is the honest answer.
- **Probe-dependence.** This is difficulty *for a linear/prototype probe*.
  Standard for frozen FM features and correlates with MLP difficulty, but state
  it. For the deployed hypothesis class, use a linear probe with the same
  matched-`k` + bootstrap trick.

**Validation it's not a frequency artifact:** sweep `k`, show the difficulty
*ranking* is stable as `k` shrinks. Stable ranking at small `k` ⇒ geometry, not
data volume.

### Richer framing: per-class learning curves

Difficulty and data-need are different axes; a single balanced point hides that.
Measure `recall_c(n)`: subsample each class to common `n`, sweep `n`, evaluate on
held-out. Fit a saturating form, e.g. `a(1 - e^{-n/τ})`:

- `a` = ceiling → **inherent difficulty** (what the class can ever reach).
- `τ` = half-saturation → **sample-efficiency / data hunger**.

Uses the whole curve, not one level. Head classes give the full curve; tail
classes give the low-`n` region, and the shared functional form lets you
extrapolate their ceiling (flag as extrapolation). Reframes the imbalance story:
"difficult-to-easy oversampling helps" becomes "oversample classes with high `a`
but large `τ`."

### Optional complement

Difficulty isn't really a scalar — a class is hard *relative to which other
class*. A C×C pairwise centroid-distance / confusability matrix (frequency-free,
symmetric prototype geometry) gives the structure behind the scalar and pairs
naturally with confusion matrices.

**Pick:** prototype recall with matched-`k` + bootstrap as the workhorse (sound,
~an afternoon); add the learning-curve `(a, τ)` decomposition if difficulty-vs-
data-need should be the headline contribution.
