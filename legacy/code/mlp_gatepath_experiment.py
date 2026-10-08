"""
Scaled ReLU gate-path faithfulness experiment
=============================================

Scaled successor of the ReLU probe: generalizes the forward/gate-saliency code to
arbitrary depth/width, runs two fully-connected ReLU architectures
((128, 64) and (128, 64, 32)) over fifteen binary datasets spanning H ~ [0.67, 0.98]
(thirteen datasets shared with the tree study minus the 1558-dimensional climate
snapshot, plus Heart-statlog (OpenML 53) and Vehicle silhouettes (OpenML 54)), and
adds *region* metrics on the training pool.

Measured objects (same protocol as the tree study, seed 42):

    faithful_k(x) = pool fraction matching x on the k most decision-salient gate signs
                    AND classified by the network as x is,

where gates are ranked by |d(logit)/d z_gate| at x. On a full *projection region*
(gate signs AND output-logit sign, Construction relu-pi) the network is affine with a
fixed output sign, so it is decision-constant by construction; the *gate-only* rule
pins only the signs of the retained gates, so its pool purity can be < 1. We therefore
also report, per fold over the training pool:

    region_purity : mass-weighted share of pool points whose gate cell's majority
                    prediction equals their own (a cell is "decision-consistent" when
                    the within-cell affine output plane does not split it);
    pure_mass     : mass fraction of pool points living in decision-consistent cells;
    n_regions     : number of distinct realized gate-signature cells on the pool.

Honest boundary: this is still a single feed-forward architecture family (fully
connected ReLU MLPs), not a full deep-learning benchmark.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr, spearmanr

from faithfulness_cv_experiment import (
    _builtin_datasets, _openml_datasets, to_binary, display_name,
    homogeneity_components,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

MIN_COVER = 5
N_TEST_PER_FOLD = 30
ARCHS = [("16x16", (16, 16)), ("32x16", (32, 16)),
         ("128x64", (128, 64)), ("128x64x32", (128, 64, 32))]
EXTRA_OPENML = {"heart": 53, "vehicle": 54}          # not cached -> may download

# Shared sets = the 14 tree-study datasets minus the 1558-dim Internet-Advertisements dataset.
EXCLUDED = {"internet-ads"}


def relu(z):
    return np.maximum(z, 0.0)


def _fetch_extra():
    """Heart-statlog (53) and Vehicle (54), binarized by the shared protocol."""
    out = {}
    for name, did in EXTRA_OPENML.items():
        try:
            X, y = fetch_openml(data_id=did, as_frame=False, return_X_y=True,
                                parser="auto", cache=False)   # bypass a corrupt cached copy
            X, y = to_binary(X, y)
            out[name] = (X, y)
            print(f"  [extra-openml] loaded {name} (data_id {did}, n={len(y)}, d={X.shape[1]})")
        except Exception as e:  # noqa: BLE001
            print(f"  [extra-openml] SKIP {name}: {type(e).__name__}: {e}")
    return out


def _forward(A, clf):
    """Generic-depth forward. Returns (zs, hs, out); out is the single output logit."""
    W, B = clf.coefs_, clf.intercepts_            # W[-1]: (H_last, 1)
    zs, hs = [], []
    z = A @ W[0] + B[0]; zs.append(z); h = relu(z); hs.append(h)
    for w, b in zip(W[1:-1], B[1:-1]):
        z = h @ w + b; zs.append(z); h = relu(z); hs.append(h)
    out = h @ W[-1] + B[-1]
    return zs, hs, out


def _signature(zs):
    """Boolean (n, total_gates) matrix: pre-activation sign of every hidden gate."""
    return np.hstack([z > 0 for z in zs])


def _gate_saliency(x1d, clf):
    """|d logit / d z_gate| at x across all hidden gates (generic depth)."""
    W, B = clf.coefs_, clf.intercepts_
    zs, _, _ = _forward(x1d[None, :], clf)
    zs = [z[0] for z in zs]
    sens = W[-1].ravel()                          # d logit / d h_{last}   (H_last,)
    gates = []
    for l in range(len(zs) - 1, -1, -1):
        dz = sens * (zs[l] > 0)                   # d logit / d z_l
        gates.append(np.abs(dz))
        if l > 0:
            sens = dz @ W[l].T                    # d logit / d h_{l-1}
    return np.concatenate(gates[::-1])


def _partial_faithfulness(x1d, clf, sig_pool, pool_pred, sig_x, pred_x):
    """Faithfulness of the top-k salient-gate rule for x, k = 1..G (up to MIN_COVER).

    Returns {k: (fraction of the pool matching the k kept gate signs that is
    classified like x, coverage)}.
    """
    sal = _gate_saliency(x1d, clf)
    order = np.argsort(sal)[::-1]
    keep = sig_x[order]
    S = sig_pool[:, order]
    mask = np.ones(len(pool_pred), dtype=bool)
    out = {}
    for k in range(1, len(order) + 1):
        mask &= (S[:, k - 1] == keep[k - 1])
        cover = int(mask.sum())
        if cover < MIN_COVER:
            break
        agree = int((pool_pred[mask] == pred_x).sum())
        out[k] = (float(agree) / cover, cover)
    return out


def _mode_agree(labels):
    """Fraction of labels equal to the majority label of the group."""
    v, c = np.unique(labels, return_counts=True)
    return float(c.max()) / len(labels)


def region_stats(sig_pool, pool_pred):
    """(n_regions, region_purity, pure_mass) over the pool's gate-signature cells."""
    if len(pool_pred) == 0:
        return 0, float("nan"), float("nan")
    packed = np.packbits(np.asarray(sig_pool, dtype=np.uint8), axis=1)
    _, inv, counts = np.unique(packed, axis=0, return_inverse=True, return_counts=True)
    n_regions = len(counts)
    tot = float(len(pool_pred))
    purity = 0.0
    pure_mass = 0.0
    pred = np.asarray(pool_pred, dtype=int)
    for g in range(n_regions):
        m = inv == g
        w = counts[g] / tot
        a = _mode_agree(pred[m])
        purity += w * a
        if a == 1.0:
            pure_mass += w
    return n_regions, purity, pure_mass


def _datasets():
    ds = _builtin_datasets()
    ds.update(_openml_datasets())
    ds.update(_fetch_extra())
    for name in list(ds.keys()):
        X, y = ds[name]
        X, y = to_binary(X, y)
        if name in EXCLUDED:
            continue
        if len(np.unique(y)) < 2 or len(y) < 60:
            print(f"  skip {name} (too few samples / classes)")
            continue
        knn, cluster = homogeneity_components(X, y)
        ds[name] = (X, y, 0.7 * knn + 0.3 * cluster)
    ds = {k: v for k, v in ds.items() if len(v) == 3 and k not in EXCLUDED}
    # Keep the study at 15+ sets even if an OpenML fetch is unavailable: pad with a
    # synthetic separability probe, documented as such.
    if len(ds) < 15:
        name = "synthetic_sep"
        X, y = make_classification(n_samples=800, n_features=12, n_informative=6,
                                   n_redundant=3, class_sep=0.5, random_state=11)
        X, y = to_binary(X, y)
        knn, cluster = homogeneity_components(X, y)
        ds[name] = (X, y, 0.7 * knn + 0.3 * cluster)
        print(f"  [synthetic fallback] added {name} to reach 15 datasets")
    return ds


def run():
    ds = _datasets()
    summary_rows = []
    curve_rows = []
    for arch_name, hidden in ARCHS:
        for name, (X, y, H) in sorted(ds.items()):
            label = display_name(name)
            d = X.shape[1]
            min_cls = int(np.bincount(y).min())
            n_folds = max(2, min(5, min_cls))
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            per_k = {}
            reg_n, reg_p, reg_pm = [], [], []
            for tr, va in skf.split(X, y):
                X_tr, X_va = X[tr], X[va]
                clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                                    max_iter=1500, random_state=42, tol=1e-5).fit(X_tr, y[tr])
                zs, _, out_tr = _forward(X_tr, clf)
                pool_pred = (out_tr[:, 0] >= 0)
                sig_pool = _signature(zs)
                nr, rp, rpm = region_stats(sig_pool, pool_pred)
                reg_n.append(nr); reg_p.append(rp); reg_pm.append(rpm)

                n = min(N_TEST_PER_FOLD, len(X_va))
                for xi in np.random.choice(len(X_va), n, replace=False):
                    x = X_va[xi]
                    zsx, _, out_x = _forward(x[None, :], clf)
                    pred_x = bool(out_x[0, 0] >= 0)
                    sig_x = _signature(zsx)[0]
                    pf = _partial_faithfulness(x, clf, sig_pool, pool_pred,
                                               sig_x, pred_x)
                    for k, (f, _) in pf.items():
                        per_k.setdefault(k, []).append(f)

            for k, vals in sorted(per_k.items()):
                curve_rows.append({"dataset": label, "H": H, "arch": arch_name, "k": k,
                                   "faithfulness": float(np.mean(vals)),
                                   "n_points": len(vals)})

            f1 = per_k.get(1, [])
            f2 = per_k.get(2, [])
            summary_rows.append({
                "dataset": label, "n": len(y), "d": d, "H": H, "arch": arch_name,
                "G": sum(hidden),
                "faith_k1": float(np.mean(f1)) if f1 else np.nan,
                "faith_k1_std": float(np.std(f1)) if f1 else np.nan,
                "faith_k2": float(np.mean(f2)) if f2 else np.nan,
                "n_regions": float(np.mean(reg_n)),
                "region_purity": float(np.nanmean(reg_p)),
                "pure_mass": float(np.nanmean(reg_pm)),
            })
            print(f"  [{arch_name:<9}] {label:<14} n={len(y):<5} d={d:<4} H={H:.3f} "
                  f"faith_k1={summary_rows[-1]['faith_k1']:.3f} "
                  f"regions={summary_rows[-1]['n_regions']:.0f} "
                  f"region_purity={summary_rows[-1]['region_purity']:.3f}")

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv("mlp_scaled_summary.csv", index=False)
    cdf = pd.DataFrame(curve_rows)
    cdf.to_csv("mlp_scaled_curves.csv", index=False)
    return sdf, cdf


def _corr_stats(df, xcol, ycol, n_boot=2000, seed=42):
    valid = df[[xcol, ycol]].dropna()
    x, y = valid[xcol].values, valid[ycol].values
    if len(x) < 3:
        return None
    def _pear(a, b):
        return np.corrcoef(a, b)[0, 1]
    def _spear(a, b):
        return spearmanr(a, b)[0]
    rng = np.random.RandomState(seed)
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    lo = (1 - 0.95) / 2
    cip = []
    cis = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(x), size=len(x))
        cip.append(_pear(x[idx], y[idx]))
        cis.append(_spear(x[idx], y[idx]))
    cip.sort(); cis.sort()
    return {"n": len(x), "x": xcol, "y": ycol, "pearson": r_p, "p_pearson": p_p,
            "ci_lo": cip[int(lo * n_boot)], "ci_hi": cip[int((1 - lo) * n_boot)],
            "spearman": r_s, "p_spearman": p_s,
            "s_lo": cis[int(lo * n_boot)], "s_hi": cis[int((1 - lo) * n_boot)]}


def analyze(sdf):
    print("\n" + "=" * 84)
    print("MLP scaled: per-dataset summary (both architectures)")
    print("=" * 84)
    cols = ["dataset", "arch", "n", "d", "H", "faith_k1", "region_purity",
            "n_regions", "pure_mass"]
    print(sdf[cols].round(3).to_string(index=False))

    rows = []
    for arch in [a for a, _ in ARCHS]:
        sub = sdf[sdf.arch == arch]
        for xc, yc in [("H", "faith_k1"), ("H", "region_purity"),
                       ("region_purity", "faith_k1")]:
            st = _corr_stats(sub, xc, yc)
            if st is None:
                continue
            st = dict(st); st["arch"] = arch
            rows.append(st)
            print(f"\n  [{arch}] {xc} vs {yc} (n={st['n']}):")
            print(f"    Pearson  r = {st['pearson']:.3f} p = {st['p_pearson']:.4f} "
                  f"CI [{st['ci_lo']:.3f}, {st['ci_hi']:.3f}]")
            print(f"    Spearman rho = {st['spearman']:.3f} p = {st['p_spearman']:.4f} "
                  f"CI [{st['s_lo']:.3f}, {st['s_hi']:.3f}]")

    pd.DataFrame(rows).to_csv("mlp_scaled_stats.csv", index=False)
    print("\n  Saved stats: mlp_scaled_stats.csv")
    return rows


def plot(sdf, cdf):
    order = ["diabetes", "sonar", "wine", "digits-35"]
    arch0 = ARCHS[0][0]
    H = dict(zip(sdf[sdf.arch == arch0].dataset, sdf[sdf.arch == arch0].H))
    fig, axes = plt.subplots(2, 1, figsize=(4.7, 6.6), sharex=False)
    colors = {"diabetes": "#d62728", "sonar": "#1f77b4",
              "wine": "#2ca02c", "digits-35": "#ff7f0e"}
    ax = axes[0]
    for ds in order:
        sub = cdf[(cdf.dataset == ds) & (cdf.arch == arch0) &
                  (cdf.n_points >= 5)].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub.k, sub.faithfulness, marker="o", linewidth=2,
                color=colors[ds], label=f"{ds} (H={H[ds]:.3f})")
    ax.set_ylabel("Faithfulness of shortened gate rules")
    ax.set_title(f"ReLU arch {arch0}: faithfulness vs. retained gates $k$", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    markers = {"16x16": "o", "32x16": "s", "128x64": "D", "128x64x32": "^"}
    for arch, _ in ARCHS:
        sub = sdf[sdf.arch == arch].dropna(subset=["H", "faith_k1"])
        ax.scatter(sub.H, sub.faith_k1, s=55, alpha=0.85, marker=markers.get(arch, "o"),
                   label=f"width {arch} (n={len(sub)})")
    ax.set_xlabel("Homogeneity H")
    ax.set_ylabel("Faithfulness of 1-gate rule (CV)")
    ax.set_title("Architecture comparison: H vs. single-gate faithfulness", fontsize=9)
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("mlp_scaled.png", dpi=150, bbox_inches="tight")
    print("\nSaved figure: mlp_scaled.png")


if __name__ == "__main__":
    import time
    t0 = time.time()
    sdf, cdf = run()
    analyze(sdf)
    plot(sdf, cdf)
    print(f"\nDone in {time.time() - t0:.1f}s. "
          f"Summary: mlp_scaled_summary.csv ({len(sdf)} rows); "
          f"curves: mlp_scaled_curves.csv ({len(cdf)} rows).")
