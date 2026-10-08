"""
Cross-validated faithfulness experiment (many datasets)
=======================================================

Hardens the partial-explanation result from `partial_explanation_experiment.py`:

  * expands from 4 to ~14 datasets (sklearn built-ins + synthetic variants + UCI via
    OpenML), spanning a range of homogeneity H and dimensionality d;
  * uses 5-fold stratified cross-validation instead of a single 70/30 split;
  * runs a sensitivity analysis on the 70/30 weighting of H = w*kNN + (1-w)*cluster.

Core quantity (same as before, "empirical sample faithfulness"):

  faithful_k(x) = fraction of TRAINING points satisfying the first-k path conditions
                  whose model prediction equals M(x).

A partial rule (first k splits) describes a region larger than the leaf; it is
faithful iff that region does not straddle a decision boundary among real data.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import fetch_openml

from xai_framework import DecisionTreeTranslator

warnings.filterwarnings("ignore")
np.random.seed(42)

DEPTHS = [2, 3, 4, 5, 6, 7, 8]
MIN_COVER = 5
N_TEST_PER_FOLD = 30


# ---------------------------------------------------------------- data loading

def _builtin_datasets():
    return {
        "iris": load_iris(return_X_y=True),
        "wine": load_wine(return_X_y=True),
        "breast_cancer": load_breast_cancer(return_X_y=True),
        "digits": load_digits(return_X_y=True),  # binarized in to_binary below
        "synthetic_d8": make_classification(
            n_samples=500, n_features=8, n_informative=5, n_redundant=2, random_state=42),
        "synthetic_d20": make_classification(
            n_samples=800, n_features=20, n_informative=10, n_redundant=4, random_state=1),
        "synthetic_d50": make_classification(
            n_samples=1000, n_features=50, n_informative=25, n_redundant=10, random_state=2),
        "synthetic_lowsep": make_classification(
            n_samples=800, n_features=10, n_informative=10, n_redundant=0,
            class_sep=0.4, random_state=3),
    }


# UCI datasets via OpenML (data_id). Tolerant: failures are skipped.
# OpenML 40978 is Internet-Advertisements (3279 instances, 1558 features).
_OPENML_IDS = {
    "banknote": 1462,      # banknote-authentication (2 classes, 4 features)
    "blood": 1464,         # blood-transfusion-service-center (2, 4)
    "diabetes": 37,        # Pima diabetes (2, 8)
    "ionosphere": 59,      # (2, 34)
    "sonar": 40,           # (2, 60)
    "internet-ads": 40978,      # Internet-Advertisements (n=3279, d=1558)
}


def _openml_datasets():
    out = {}
    for name, did in _OPENML_IDS.items():
        try:
            X, y = fetch_openml(data_id=did, as_frame=False, return_X_y=True,
                                parser="auto")
            out[name] = (X, y)
            print(f"  [openml] loaded {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  [openml] SKIP {name}: {type(e).__name__}: {e}")
    return out


def to_binary(X, y):
    """Standardize + binarize to the two most frequent classes; return (X, y).

    For load_digits the surviving pair is classes {3, 5} (183 + 182 = 365 samples,
    NOT the literal {0, 1} classes, which number only 360); the two most frequent
    classes are used throughout for a uniform binary protocol.
    """
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        X = np.nan_to_num(X)  # some datasets have missing values
    y = pd.factorize(np.asarray(y, dtype=str))[0]
    if len(np.unique(y)) > 2:
        counts = np.bincount(y)
        top2 = np.argsort(counts)[-2:]
        mask = np.isin(y, top2)
        X, y = X[mask], (y[mask] == top2[1]).astype(int)
    X = StandardScaler().fit_transform(X)
    return X, y


def display_name(name):
    """CSV / figure label for a dataset key (digits -> digits-35)."""
    return "digits-35" if name == "digits" else name


# ---------------------------------------------------------------- metrics

def homogeneity_components(X, y, k=5, n_clusters=5):
    """Return (kNN consistency, cluster purity); H = w*knn + (1-w)*cluster."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X)
    _, indices = nbrs.kneighbors(X)
    knn = float(np.mean([np.mean(y[indices[i][1:]] == y[i]) for i in range(len(y))]))

    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)
    purities = []
    for c in range(n_clusters):
        m = (labels == c)
        if m.sum() > 0:
            purities.append(float(np.bincount(y[m]).max() / m.sum()))
    cluster = float(np.mean(purities))
    return knn, cluster


def _prefix_mask(path_prefix, pool_X):
    mask = np.ones(len(pool_X), dtype=bool)
    for feat, direction, thresh in path_prefix:
        if direction == "leq":
            mask &= pool_X[:, feat] <= thresh
        else:
            mask &= pool_X[:, feat] > thresh
    return mask


def partial_faithfulness(translator, tree, x, pool_X, min_cover=MIN_COVER):
    """{k: (faithfulness, coverage)} for each prefix rule length k of input x."""
    path, pred, _ = translator.find_activated_path(x)
    L = len(path)
    out = {}
    for k in range(1, L + 1):
        sat = _prefix_mask(path[:k], pool_X)
        cover = int(sat.sum())
        if cover < min_cover:
            out[k] = (float("nan"), cover)
        else:
            preds = tree.predict(pool_X[sat])
            out[k] = (float(np.mean(preds == pred)), cover)
    return out


# ---------------------------------------------------------------- main

def run():
    datasets = _builtin_datasets()
    datasets.update(_openml_datasets())

    rows = []
    kcurves = []   # per (dataset, k): faithfulness across folds x depths x test samples
    for name, (X, y) in datasets.items():
        label = display_name(name)
        X, y = to_binary(X, y)
        if len(np.unique(y)) < 2 or len(y) < 60:
            print(f"  skip {name} (too few samples / classes)")
            continue

        knn, cluster = homogeneity_components(X, y)
        d = X.shape[1]

        min_cls = int(np.bincount(y).min())
        n_folds = max(2, min(5, min_cls))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fs_k1, fs_k2 = [], []
        per_k = {}   # k -> list of faithfulness values for this dataset
        for tr, va in skf.split(X, y):
            X_tr, X_va, y_tr = X[tr], X[va], y[tr]
            for depth in DEPTHS:
                tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_tr, y_tr)
                trans = DecisionTreeTranslator(tree, X_train=X_tr)
                n = min(N_TEST_PER_FOLD, len(X_va))
                for xi in np.random.choice(len(X_va), n, replace=False):
                    pf = partial_faithfulness(trans, tree, X_va[xi], X_tr)
                    for k, (f, cov) in pf.items():
                        if np.isnan(f):
                            continue
                        per_k.setdefault(k, []).append(f)
                        if k == 1:
                            fs_k1.append(f)
                        elif k == 2:
                            fs_k2.append(f)

        for k, vals in sorted(per_k.items()):
            kcurves.append({
                "dataset": label, "H": 0.7 * knn + 0.3 * cluster, "k": k,
                "faithfulness": float(np.mean(vals)), "n_points": len(vals),
            })

        rows.append({
            "dataset": label, "n": len(y), "d": d,
            "knn": knn, "cluster": cluster, "H": 0.7 * knn + 0.3 * cluster,
            "faith_k1": np.mean(fs_k1) if fs_k1 else np.nan,
            "faith_k1_std": np.std(fs_k1) if fs_k1 else np.nan,
            "faith_k2": np.mean(fs_k2) if fs_k2 else np.nan,
        })
        print(f"  {label:<16} n={len(y):<5} d={d:<3} H={0.7*knn+0.3*cluster:.3f} "
              f"faith_k1={rows[-1]['faith_k1']:.3f}")

    df = pd.DataFrame(rows).sort_values("H")
    df.to_csv("faithfulness_cv_results.csv", index=False)

    kcurves_df = pd.DataFrame(kcurves)
    kcurves_df.to_csv("faithfulness_vs_k_cv.csv", index=False)
    return df, kcurves_df


def bootstrap_ci(x, y, stat, n_boot=2000, seed=42, ci=0.95):
    """Percentile bootstrap CI for stat(x, y) over paired resamples (n = #datasets)."""
    rng = np.random.RandomState(seed)
    lo = (1 - ci) / 2
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        vals.append(stat(x[idx], y[idx]))
    vals = np.sort(vals)
    return vals[int(lo * n_boot)], vals[int((1 - lo) * n_boot)]


def analyze(df):
    from scipy.stats import spearmanr, pearsonr

    print("\n" + "=" * 82)
    print("All datasets (sorted by H)")
    print("=" * 82)
    cols = ["dataset", "n", "d", "H", "faith_k1", "faith_k1_std", "faith_k2"]
    print(df[cols].round(3).to_string(index=False))

    valid = df["faith_k1"].notna()
    Hv = df["H"][valid].values
    Fv = df["faith_k1"][valid].values

    def _pear(a, b):
        return np.corrcoef(a, b)[0, 1]

    def _spear(a, b):
        return spearmanr(a, b)[0]

    r_p, p_p = pearsonr(Hv, Fv)
    r_s, p_s = spearmanr(Hv, Fv)
    ci_p = bootstrap_ci(Hv, Fv, _pear)
    ci_s = bootstrap_ci(Hv, Fv, _spear)
    print("\n  headline (w=0.7):")
    print(f"    Pearson  r = {r_p:.3f}  p = {p_p:.4f}   95% bootstrap CI [{ci_p[0]:.3f}, {ci_p[1]:.3f}]")
    print(f"    Spearman rho = {r_s:.3f}  p = {p_s:.4f}   95% bootstrap CI [{ci_s[0]:.3f}, {ci_s[1]:.3f}]")
    print(f"    (n = {int(valid.sum())} datasets, 2000 resamples, seed 42)")

    print("\n" + "=" * 82)
    print("Sensitivity of H = w*kNN + (1-w)*cluster  (correlation with faith_k1)")
    print("=" * 82)
    print(f"{'w':>5} {'Pearson r':>11} {'p':>9} {'CI_lo':>7} {'CI_hi':>7} {'Spearman rho':>13}")
    results = {}
    stat_rows = []
    for w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        Hw = (w * df["knn"] + (1 - w) * df["cluster"])[valid].values
        r, p = pearsonr(Hw, Fv)
        rho, _ = spearmanr(Hw, Fv)
        lo, hi = bootstrap_ci(Hw, Fv, _pear)
        results[w] = (r, rho)
        stat_rows.append({"w": w, "pearson_r": r, "p": p, "ci_lo": lo, "ci_hi": hi,
                          "spearman_rho": rho})
        print(f"{w:>5.1f} {r:>11.3f} {p:>9.4f} {lo:>7.3f} {hi:>7.3f} {rho:>13.3f}")

    stats_df = pd.DataFrame([{
        "metric": "headline_w0.7_pearson", "value": r_p, "p": p_p, "ci_lo": ci_p[0], "ci_hi": ci_p[1],
    }, {
        "metric": "headline_w0.7_spearman", "value": r_s, "p": p_s, "ci_lo": ci_s[0], "ci_hi": ci_s[1],
    }] + [{
        "metric": f"w{s['w']:.1f}_pearson", "value": s["pearson_r"], "p": s["p"],
        "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
    } for s in stat_rows])
    stats_df.to_csv("faithfulness_cv_stats.csv", index=False)
    print("\n  Saved stats: faithfulness_cv_stats.csv")
    return results


def plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    valid = df[df.faith_k1.notna()]
    ax = axes[0]
    for _, row in valid.iterrows():
        ax.scatter(row.H, row.faith_k1, s=100 + 12 * row.d,
                   alpha=0.8, edgecolor="k", linewidth=0.4)
        ax.annotate(row.dataset, (row.H, row.faith_k1),
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xlabel("Homogeneity H (w = 0.7)")
    ax.set_ylabel("Faithfulness at k = 1 (mean over CV folds & depths)")
    ax.set_title("Shallow-rule faithfulness vs. H (point size ~ dimensionality d)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    from scipy.stats import spearmanr
    ws_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pears, spear = [], []
    for w in ws_list:
        Hw = w * df["knn"] + (1 - w) * df["cluster"]
        v = df["faith_k1"].notna()
        pears.append(np.corrcoef(Hw[v], df["faith_k1"][v])[0, 1])
        spear.append(spearmanr(Hw[v], df["faith_k1"][v])[0])
    ax.plot(ws_list, pears, marker="o", label="Pearson r")
    ax.plot(ws_list, spear, marker="s", label="Spearman rho")
    ax.set_xlabel("weight w on kNN term in H")
    ax.set_ylabel("Correlation with faith_k1")
    ax.set_title("Sensitivity of the result to the H weighting")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("faithfulness_cv.png", dpi=150, bbox_inches="tight")
    print("\nSaved figure: faithfulness_cv.png")


def plot_kcurves(kcurves_df):
    """Faithfulness vs rule length k (CV-aggregated), four datasets spanning H.

    Replaces the old 4-dataset non-CV partial_faithfulness.png panel: this curve is
    averaged over the same 5-fold CV / depth-2..8 grid as the headline r=0.69 result,
    and carries NO correlation annotation.
    """
    order = ["diabetes", "sonar", "wine", "digits-35"]
    colors = {"diabetes": "#d62728", "sonar": "#1f77b4",
              "wine": "#2ca02c", "digits-35": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for ds in order:
        sub = kcurves_df[kcurves_df.dataset == ds].sort_values("k")
        ax.plot(sub.k, sub.faithfulness, marker="o", linewidth=2,
                color=colors[ds], label=f"{ds} (H={sub.H.iloc[0]:.3f})")
    ax.set_xlabel("Rule length k (number of path conditions kept)")
    ax.set_ylabel("Faithfulness (CV-aggregated model agreement)")
    ax.set_title("Faithfulness of shortened rules vs. rule length (5-fold CV, depths 2-8)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("faithfulness_vs_k_cv.png", dpi=150, bbox_inches="tight")
    print("\nSaved figure: faithfulness_vs_k_cv.png")


if __name__ == "__main__":
    import time
    t0 = time.time()
    df, kcurves_df = run()
    analyze(df)
    plot(df)
    plot_kcurves(kcurves_df)
    print(f"\nDone in {time.time() - t0:.1f}s. "
          f"Results: faithfulness_cv_results.csv ({len(df)} datasets), "
          f"faithfulness_vs_k_cv.csv ({len(kcurves_df)} rows).")
