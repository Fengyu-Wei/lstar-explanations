# Locked protocol: independent-pool rule utility

Date: 2026-10-05. CONFIRMATORY unless explicitly marked exploratory.
Question: Does shortening a fitted tree's explanation reduce the number of conditions and increase rule reuse without hiding fidelity loss?

## Data and independence
Use the original fourteen binary datasets, retaining original synthetic generators and the two most frequent classes. Read OpenML files from the existing local cache (no silent skipping). Seeds 0--9; maximum tree depths 4 and 8. For each seed, stratified four-way split: 50% fit, 20% select, 25% independent reference, 5% query (up to 20 queries). Fit median imputation only on fit data; tree does not need scaling. Every method explains the same queries and is evaluated on the same reference data. No tuning on reference data. Save split indices, model accuracy, class frequencies, all per-query results, and runtimes. Zero-reference coverage means undefined precision, never a perfect score or discarded query for length/coverage summaries.

## Methods
1. Full decision path (exact baseline).
2. Earliest structurally pure prefix (exact, prefix restricted).
3. Reverse-order greedy deletion of path literals, verified by intersection with all opposite-class leaf boxes (exact, subset-minimal within path literals; not minimum-feature AXp).
4. Shortest empirical prefix: select precision >= .95 and count >= 5; full-path fallback.
5. Greedy empirical path-literal deletion: same select criterion, longest deletable suffix literal first, full-path start.
6. Certified prefix: same candidate prefix family; one-sided exact binomial lower bounds with delta=.05 divided by 2*number-of-tree-nodes, fixed model-wide node/class family; structural purity is also accepted. Select earliest passing prefix; full path always available. This is a standard statistical safeguard, not a novelty claim. Assumption: iid future data from calibration distribution, trained tree fixed independently of calibration. Non-monotone precision means scan all prefixes.
7. Root/constant rule (imbalance control, not a proposed shortening method).

## Locked endpoints
Equal-weight dataset macro means: number of literals, reference coverage fraction, conditional model precision on rules with nonzero reference count, query evaluability rate, fraction of query rules whose measured precision < .95, fraction shortened vs full path. Report class-balanced conditional precision as additional endpoint, separately report ground-truth accuracy of fitted models. Low precision on finite reference data is an empirical diagnostic, not proof a statistical guarantee failed. Bootstrap intervals: 5000 resamples, seed 20261005, resample datasets and within-dataset paired seeds, preserving method pairing; CIs reflect this benchmark/resampling scheme, not deployment guarantees. Compare exact methods on paired length/coverage differences. Timing excludes fitting and reference evaluation; certify node tables amortized over queries and report construction separately. No claims of human usability from literal counts alone.

## Official Anchors comparison
Use anchor-exp 0.0.2.0, lime 0.2.0.1, default quartile discretization, target .95, delta .05, tau .05, batch 100, maximum anchor size 8, coverage_samples=1000. Fit its discretizer/sampler on fit data only. Preselected moderate-sized benchmarks: breast_cancer, banknote, blood, diabetes, wine, digits-35, sonar. Seeds 0--2, depth 8, first five query indices. Evaluate returned rules on the SAME unperturbed reference pool and report native sampling precision separately. Preserve every failure and unmet target. Count atomic inequalities, noting their thresholds/family differ from tree paths. Compare other methods on precisely this matched subset; do not generalize restricted search outcomes to unrestricted Anchors. No timed-out query may be silently excluded.

## Paired curves and sensitivity
For every query use prefix length min(k,L), k=1..8; hold the contributing query set fixed. Save counts and missingness. Threshold sensitivity .90/.95/.99 and selection-count sensitivity 5/10/20 reuse fixed fits; report as prespecified secondary analysis, not choose best afterwards.

## Practical cases
For breast_cancer and banknote, seed 0/depth8, choose longest full path among query inputs (tie: lowest original index). Export exact and approximate rules, model vs label outcome, all supports and independent-reference precision. Cases illustrate audit/reuse only, not clinical or financial usefulness validation.

## Hypotheses and success criteria
H1: Exact condition deletion has lower macro length and nondecreasing coverage vs full paths; support iff paired length-difference CI <0 and verified fidelity checks pass. Earliest pure prefixes may show little gain: record this rather than hide it.
H2: Empirical prefixes improve length/coverage but sometimes miss the .95 held-out target; certified prefixes trade some shortening for reliable evidence. Support measured separately, no requirement to make certified method win. Report small-calibration failure to shorten as a negative result.
H3: Rule reuse can be quantified without calling it human understanding: covered-instance counts, condition reduction, runtime and readable case exports.

## Cross-model robustness follow-up (separate protocol required)
Re-fit logistic regression and a 16x16 ReLU model with fit-only scaling and independent reference pools. Do not claim the new tree utility method generalizes to those families.
