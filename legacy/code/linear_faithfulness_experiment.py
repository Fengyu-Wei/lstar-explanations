"""
Linear-model faithfulness experiment
====================================

Extends the tree study of `faithfulness_cv_experiment.py` to *linear models*, giving
empirical coverage for the linear half of the paper's claim ("Linear and Tree Models").
Same datasets, same 5-fold CV protocol and seed as the tree experiment, so the
dataset-level homogeneity values H are identical to `faithfulness_cv_results.csv`.

Faithfulness analog for a logistic-regression decision boundary.

For a test input x with full-model coefficients w and intercept b, order the features
by the magnitude of their *local* signed contribution |w_i x_i| (the `local` ranking)
or, as a robustness check, by the input-independent coefficient magnitude |w_i| (the
`coef` ranking). Keeping only the top-s features defines a shortened linear rule (the
reduced decision hyperplane on those coordinates). Its region is

    R_s(x) = { x' : sign(w_S . x'_S + b) = sign(w_S . x_S + b) },   S = top-s features,

which is exactly analogous to the tree prefix region (a rule over the kept conditions).
As in the tree study, faithfulness is the fraction of *training* points inside R_s(x)
on which the FULL model prediction equals the full prediction at x:

    faithful_s(x) = |{ x' in R_s(x) : M(x') = M(x) }| / |R_s(x)|.

When s equals the full feature set, R_s is a whole halfspace on which M is constant, so
faithfulness is 1 by construction -- the linear analogue of the tree's complete path.
The interesting quantity is the shortened rule s = 1 (and s = 2): can a single most
influential feature sustain the model's decision on real data, and does dataset
homogeneity H predict that it can?

We report the dataset-level correlation of H with the s = 1 faithfulness (Pearson /
Spearman with p-values and a 2000-resample bootstrap 95% CI), mirroring the tree
result, under *both* rankings so that the (weak) linear signal can be checked for
dependence on the feature-ordering choice.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr, spearmanr

from faithfulness_cv_experiment import (
    _builtin_datasets, _openml_datasets, to_binary, display_name,
    homogeneity_components,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

MIN_COVER = 5
N_TEST_PER_FOLD = 100
S_VALUES = [1, 2]
ZERO_EPS = 1e-12
RANKINGS = ["local", "coef"]


def linear_partial_faithfulness(w, b, x, s, pool_X, full_preds, rank="local"):
    """Faithfulness of the rule that keeps the top-s features of the shortened rule.

    ``rank='local'`` orders features by |w_i x_i| (magnitude of the signed contribution
    at x); ``rank='coef'`` orders them by |w_i| (input-independent coefficient size).
    Returns (faithfulness, coverage) or (nan, coverage) if the region has < MIN_COVER
    training points.
    """
    if rank == "local":
        magnitude = np.abs(w * x)
    else:
        magnitude = np.abs(w)
    order = np.argsort(magnitude)[::-1]
    order = order[magnitude[order] > ZERO_EPS]
    if len(order) < s:
        return float("nan"), 0
    feats = order[:s]

    red = pool_X[:, feats] @ w[feats] + b            # reduced decision at each pool point
    target_sign = (x[feats] @ w[feats] + b) >= 0     # reduced decision at x
    region = ((red >= 0) == target_sign)
    cover = int(region.sum())
    if cover < MIN_COVER:
        return float("nan"), cover
    # full prediction of the model at x:
    fx = bool((w @ x + b) >= 0)
    agree = (full_preds[region] == fx)
    return float(agree.mean()), cover


def run():
    datasets = _builtin_datasets()
    datasets.update(_openml_datasets())

    rows = {r: [] for r in RANKINGS}
    for name, (X, y) in datasets.items():
        label = display_name(name)
        X, y = to_binary(X, y)
        if len(np.unique(y)) < 2 or len(y) < 60:
            print(f"  skip {name} (too few samples / classes)")
            continue

        knn, cluster = homogeneity_components(X, y)
        H = 0.7 * knn + 0.3 * cluster
        d = X.shape[1]

        min_cls = int(np.bincount(y).min())
        n_folds = max(2, min(5, min_cls))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        per_s = {r: {s: [] for s in S_VALUES} for r in RANKINGS}
        for tr, va in skf.split(X, y):
            X_tr, X_va, y_tr = X[tr], X[va], y[tr]
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(X_tr, y_tr)
            w = clf.coef_.ravel()
            b = float(clf.intercept_[0])
            full_preds = (X_tr @ w + b) >= 0          # full-model decision on the pool
            n = min(N_TEST_PER_FOLD, len(X_va))
            for xi in np.random.choice(len(X_va), n, replace=False):
                x = X_va[xi]
                for r in RANKINGS:
                    for s in S_VALUES:
                        f, _ = linear_partial_faithfulness(w, b, x, s, X_tr,
                                                           full_preds, rank=r)
                        if not np.isnan(f):
                            per_s[r][s].append(f)

        for r in RANKINGS:
            rows[r].append({
                "dataset": label, "n": len(y), "d": d, "H": H,
                "faith_s1": np.mean(per_s[r][1]) if per_s[r][1] else np.nan,
                "faith_s1_std": np.std(per_s[r][1]) if per_s[r][1] else np.nan,
                "faith_s2": np.mean(per_s[r][2]) if per_s[r][2] else np.nan,
            })
        print(f"  {label:<16} n={len(y):<5} d={d:<3} H={H:.3f} "
              f"faith_s1[local]={rows['local'][-1]['faith_s1']:.3f} "
              f"faith_s1[coef]={rows['coef'][-1]['faith_s1']:.3f}")

    frames = {}
    for r in RANKINGS:
        frames[r] = pd.DataFrame(rows[r]).sort_values("H")
    frames["local"].to_csv("linear_faithfulness_results.csv", index=False)
    frames["coef"].to_csv("linear_faithfulness_coef_results.csv", index=False)
    return frames


def analyze(df, tag=""):
    """Correlation of H with the s=1 faithfulness for one ranking's dataframe."""
    print("\n" + "=" * 82)
    print(f"Linear model ({tag or 'local'} ranking): all datasets (sorted by H)")
    print("=" * 82)
    cols = ["dataset", "n", "d", "H", "faith_s1", "faith_s1_std", "faith_s2"]
    print(df[cols].round(3).to_string(index=False))

    valid = df["faith_s1"].notna()
    Hv = df["H"][valid].values
    Fv = df["faith_s1"][valid].values

    def _pear(a, b):
        return np.corrcoef(a, b)[0, 1]

    def _spear(a, b):
        return spearmanr(a, b)[0]

    def bootstrap_ci(x, y, stat, n_boot=2000, seed=42, ci=0.95):
        rng = np.random.RandomState(seed)
        lo = (1 - ci) / 2
        vals = []
        n = len(x)
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            vals.append(stat(x[idx], y[idx]))
        vals = np.sort(vals)
        return vals[int(lo * n_boot)], vals[int((1 - lo) * n_boot)]

    r_p, p_p = pearsonr(Hv, Fv)
    r_s, p_s = spearmanr(Hv, Fv)
    ci_p = bootstrap_ci(Hv, Fv, _pear)
    ci_s = bootstrap_ci(Hv, Fv, _spear)
    print(f"\n  linear headline ({tag} ranking, H vs faith_s1):")
    print(f"    Pearson  r = {r_p:.3f}  p = {p_p:.4f}   95% bootstrap CI [{ci_p[0]:.3f}, {ci_p[1]:.3f}]")
    print(f"    Spearman rho = {r_s:.3f}  p = {p_s:.4f}   95% bootstrap CI [{ci_s[0]:.3f}, {ci_s[1]:.3f}]")
    print(f"    (n = {int(valid.sum())} datasets, 2000 resamples, seed 42)")

    fname = "linear_faithfulness_stats.csv" if tag == "local" \
        else "linear_faithfulness_coef_stats.csv"
    pd.DataFrame([{
        "metric": f"linear_faith_s1_pearson_{tag}", "value": r_p, "p": p_p,
        "ci_lo": ci_p[0], "ci_hi": ci_p[1],
    }, {
        "metric": f"linear_faith_s1_spearman_{tag}", "value": r_s, "p": p_s,
        "ci_lo": ci_s[0], "ci_hi": ci_s[1],
    }]).to_csv(fname, index=False)
    print(f"  Saved stats: {fname}")
    return r_p, p_p, r_s, p_s, ci_p, ci_s


def plot(frames):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), sharey=True)
    titles = {"local": "Ordered by |w_i x_i| (local contribution)",
              "coef": "Ordered by |w_i| (coefficient magnitude)"}
    for ax, r in zip(axes, RANKINGS):
        df = frames[r]
        valid = df[df.faith_s1.notna()]
        for _, row in valid.iterrows():
            ax.scatter(row.H, row.faith_s1, s=90 + 10 * row.d, alpha=0.85,
                       edgecolor="k", linewidth=0.4)
            ax.annotate(row.dataset, (row.H, row.faith_s1),
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xlabel("Homogeneity H (w = 0.7)")
        ax.set_title(titles[r], fontsize=10)
        ax.set_xlim(0.55, 1.02)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Faithfulness of 1-feature linear rule (CV)")
    plt.tight_layout()
    plt.savefig("linear_faithfulness.png", dpi=150, bbox_inches="tight")
    print("\nSaved figure: linear_faithfulness.png (two ranking panels)")


if __name__ == "__main__":
    import time
    t0 = time.time()
    frames = run()
    stats = {}
    for r in RANKINGS:
        stats[r] = analyze(frames[r], tag=r)
    plot(frames)
    print(f"\nDone in {time.time() - t0:.1f}s. "
          f"Results: linear_faithfulness_results.csv, "
          f"linear_faithfulness_coef_results.csv ({len(frames['local'])} datasets each).")
