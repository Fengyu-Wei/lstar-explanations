# Reviewer follow-up: compression diagnostics and region controls

Date: 2026-10-07. User authorized supplementary experiments and requested GPU execution.
These are post-review analyses, not original preregistered endpoints. Earlier results,
including a preliminary reference-pool mass-ratio summary and matched reliability
comparison, have already been inspected. No reference-based tuning is permitted.

## Frozen inputs and computation

Reuse the original fourteen tasks, corrected four-way splits, seeds 0--9 and depths
4/8 for trees; reuse the stored 65 neural fits/queries (13 tasks, seeds 0--4).
Hash all input result files. Keep all original result files unchanged.
Original CART fits are reconstructed with the original CPU scikit-learn implementation
because switching training algorithms would change the fitted classifiers. GPU float64
computes tree routing, candidate-region counts, sibling losses and hierarchical
bootstrap resampling. Numerical GPU results must reproduce stored CPU results.
Neural region controls use stored counts from exactly the same previously audited fits;
no new neural training or model selection is required. Record GPU and runtime versions.

## A. Mass budget and sibling diagnostics

Evaluate all ordered prefixes, including root and full path. For the descriptive
budget study, use distinct nonempty proper prefixes 1 <= k < L with reference support.
Report empirical q = leaf support / prefix support, observed precision, precision-q,
budget reaching .95, and structurally sufficient prefixes missed by that mass budget.
Zero leaf support makes q=0 and the mass budget vacuous. Zero prefix support makes
precision/q undefined. Query-average candidates first, then seed and task, to avoid
weighting longer paths more heavily. Retain eligibility rates and raw candidates.

Add a calibration mass-budget prefix control: choose the shortest prefix with at least
five calibration points and calibration q >= .95, else full path. This is a finite-pool
acceptance rule, NOT a population certificate. Include root as in the original empirical
prefix. Compare on identical queries with full path, pure prefix and empirical prefix.

Sibling disagreement counts telescope to each prefix's precision deficit. Check this
identity and export every component. This identity check is implementation validation,
not independent evidence for predictive usefulness. For independent localization,
at each nonempty proper prefix with at least two sibling regions having >=5 calibration
points and some calibration disagreements, select the sibling with most calibration
disagreements (tie: earliest path position). On reference data, report its share of all
prefix disagreements, its hit on a maximally disagreeing sibling, and its excess share
over uniform selection among the same calibration-eligible siblings. Shares/hits are
conditional on nonzero reference disagreements; retain all eligible zero-error cases.
This diagnoses loss sources, not a new globally optimal compression algorithm.
Illustrative paths are chosen by longest full path at seed 0/depth8, lowest source
index on ties, for Breast Cancer and Banknote, matching the existing case policy.

## B. Same-network representation controls

Join stored complete-projection queries and approximate gate queries one-to-one.
Compare predicted-class region D_c, full activation cell plus output comparison,
all gates without output, and 1/2/4/8 retained gates. D_c is a semantic reference,
not a claim of concise human explanation. Its support equals the stored query-class
prediction frequency times reference pool size. Validate integer support recovery and
matching labels, full-cell support <= all-gate support, and complete support <= D_c.
Report paired coverage/evaluability differences on all common queries and retain
undefined precision. Do not interpret activation-cell sparsity as intrinsic to all
exact neural explanation representations.

## C. Matched reliability uncertainty

Use the existing 105 Anchors-matched queries; compare minimum empirical subset and
greedy empirical deletion. Compute per-query paired target-miss differences and
precision-shortfall differences max(0,.95-precision) only when both are supported.
Report support eligibility, task/seed macro means, paired 95% hierarchical bootstrap
intervals, and all seven leave-one-task-out summaries, including exclusion of digits-35.
Do not expand the sample or choose seeds based on significance.

## Uncertainty and outputs

Use 5,000 dataset/within-dataset seed resamples, RNG seed 20261007. Resample methods
jointly, and keep queries paired within seed. Intervals describe benchmark variability,
not independent deployment trials. GPU computations are checked against CPU reference
means/intervals, without calling CPU results additional experimental evidence.
Save per-query/prefix/sibling data, macro estimates/intervals, input hashes, runtime
metadata, readable cases, figures, and manuscript-ready tables in this new directory.
Integrate results honestly, including conservative bounds or inconclusive differences.
