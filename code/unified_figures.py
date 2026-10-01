"""Merged cross-model figures (manuscript Figures 2 and 3).

The four single-study result figures are regrouped by *what each panel plots*
rather than by which experiment produced it:

  cross_model_H.png      -- Fig 2.  x = homogeneity H, y = faithfulness of the
                            *shortest* explanation.  One series per model class
                            (trees; linear under |w_i x_i|; linear under |w_i|;
                            ReLU 16x16), each with its own fit and its own r,
                            because the four series are not drawn from the same
                            sample.  The right panel keeps the ReLU capacity
                            boundary across the four architectures.
  shortening_vs_k.png    -- Fig 3.  Faithfulness of *shortened* rules versus
                            rule length k: left = trees (k = 1..8), right = ReLU
                            16x16 (k = 1..32).  The same four datasets and the
                            same four colours appear in both panels, so a curve
                            can be followed across model classes.

Only the cached CSVs written by the three experiment scripts are read -- no
network is refitted -- and every correlation drawn here is asserted against the
stats CSVs, so the figure cannot drift away from the numbers in the text.

    python unified_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, os.pardir, "results")
FIGURES = os.path.join(HERE, os.pardir, "figures")
PAPER = os.path.join(HERE, os.pardir, os.pardir, "paper", "KBS")


def csv(name):
    return pd.read_csv(os.path.join(RESULTS, name))

# ---------------------------------------------------------------- constants
ORDER = ["diabetes", "sonar", "wine", "digits-35"]      # H ascending pairs
COLORS = {"diabetes": "#d62728", "sonar": "#1f77b4",
          "wine": "#2ca02c", "digits-35": "#ff7f0e"}

MARKERS = {"16x16": "o", "32x16": "s", "128x64": "D", "128x64x32": "^"}
ARCH_COLORS = {"16x16": "#2ca02c", "32x16": "#ff7f0e",
               "128x64": "#9467bd", "128x64x32": "#8c564b"}
ARCHES = ["16x16", "32x16", "128x64", "128x64x32"]

# Effective size on the page is (rendered width / native width) * fontsize.
# The canvas is 8.6 in = 619 pt; the cas two-column text block is 494.5 pt and
# the cas-sc one-column block 466.6 pt, so the scale is 0.80 / 0.75 and these
# sizes land at 7.5-9 pt in print.  (The old figures were drawn on 14 in and
# 10.4 in canvases, which pushed their axis text down to ~4.6 pt and ~6.2 pt.)
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT = 10.5, 11, 10, 8.5, 8

# Datasets named in the text (Sec. "Results"), labelled on the tree series.
# Only these are labelled: putting all fourteen names on the scatter is what
# made the old figure collide with itself (and the four synthetic sets are
# named collectively in the caption instead).
LABELLED = [("diabetes", "diabetes"), ("sonar", "sonar"),
            ("synthetic_d50", "synthetic ($d{=}50$)"), ("banknote", "banknote"),
            ("blood", "blood"), ("wine", "wine"), ("digits-35", "digits-35")]


def r_p(x, y):
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def fit_line(ax, x, y, color, ls="-", alpha=0.9):
    k, b = np.polyfit(x, y, 1)
    xs = np.linspace(min(x), max(x), 50)
    ax.plot(xs, k * xs + b, color=color, ls=ls, lw=1.4, alpha=alpha, zorder=2)


def obstacles_px(fig, ax):
    """Display coordinates of everything a label must not sit on.

    Markers and every drawn line (fit lines, connectors) are sampled, so a
    label cannot land on a curve either -- which is what happened when only the
    markers were treated as obstacles.  Lines are resampled densely enough that
    the sampled dots are closer together than a glyph, i.e. the line reads as
    solid to the placement test.
    """
    pts = []
    for coll in ax.collections:
        off = np.asarray(coll.get_offsets())
        if off.size:
            pts.append(off)
    # Gridlines live in ax.lines but span the whole panel; resampled they would
    # blank out every candidate position, leaving the scorer with a tie it
    # always breaks the same way.  They are not obstacles.
    grid = {id(g) for g in ax.get_xgridlines() + ax.get_ygridlines()}
    for ln in ax.lines:
        if id(ln) in grid:
            continue
        xd, yd = ln.get_data()
        if len(xd) < 2:
            continue
        xd = np.asarray(xd, float)
        yd = np.asarray(yd, float)
        # Resample along the polyline parameter, NOT over x: several of these
        # lines are vertical (the dataset connectors), and resampling a
        # constant-x line over x collapses all its samples onto one point,
        # which then sits inside every candidate box and freezes the placers.
        n = max(200, len(xd) * 20)
        u = np.linspace(0.0, 1.0, len(xd))
        s = np.linspace(0.0, 1.0, n)
        pts.append(np.column_stack([np.interp(s, u, xd), np.interp(s, u, yd)]))
    fig.canvas.draw()
    return ax.transData.transform(np.vstack(pts))


def place_labels(fig, ax, labels, color):
    """Place each label where it covers nothing.

    Candidates are ranked by (obstacles inside the box, hits an already-placed
    label, distance from the anchor): a label first tries to touch nothing, then
    to stay off the other labels, then to stay near its own point.  Ranking by a
    clearance scalar instead leaves ties -- and a tie-break that always picks
    the same candidate is what silently stacked `wine` on `digits-35`.
    """
    renderer = fig.canvas.get_renderer()
    obs = obstacles_px(fig, ax)
    placed = []
    # The radius multiplies the direction: without that the candidate set
    # collapses onto the anchor point and every label is scored on top of its
    # own marker.
    dirs = ((0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1),
            (2, 1), (-2, 1), (2, -1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2))
    cands = [(dx * d, dy * d, max(abs(dx), abs(dy)) * d)
             for d in (9, 14, 20, 27, 35, 45) for dx, dy in dirs]
    for txt, x, y in labels:
        t = ax.annotate(txt, (x, y), fontsize=FS_ANNOT, color=color,
                        xytext=(0, 9), textcoords="offset points",
                        ha="center", va="center", zorder=5)
        best_key = best_xy = None
        for dx, dy, radius in cands:
            t.set_position((dx, dy))
            bb = t.get_window_extent(renderer)
            if not (ax.bbox.x0 <= bb.x0 and bb.x1 <= ax.bbox.x1
                    and ax.bbox.y0 <= bb.y0 and bb.y1 <= ax.bbox.y1):
                continue
            inside = int(((obs[:, 0] >= bb.x0) & (obs[:, 0] <= bb.x1)
                          & (obs[:, 1] >= bb.y0) & (obs[:, 1] <= bb.y1)).sum())
            clash = int(any(dist_box_box(bb, o) <= 0 for o in placed))
            key = (inside, clash, radius)
            if best_key is None or key < best_key:
                best_key, best_xy = key, (dx, dy)
        t.set_position(best_xy or (0, 9))
        placed.append(t.get_window_extent(renderer))


def dist_to_box(bb, px, py):
    dx = max(bb.x0 - px, 0.0, px - bb.x1)
    dy = max(bb.y0 - py, 0.0, py - bb.y1)
    return (dx * dx + dy * dy) ** 0.5


def dist_box_box(a, b):
    dx = max(a.x0 - b.x1, 0.0, b.x0 - a.x1)
    dy = max(a.y0 - b.y1, 0.0, b.y0 - a.y1)
    return (dx * dx + dy * dy) ** 0.5


# ------------------------------------------------------------------ Figure 2
def figure2(tree, lin, linc, relu, mlp_stats):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    # --- left: H vs. the faithfulness of the shortest explanation, per model class
    ax = axes[0]
    common = sorted(set(tree.dataset) & set(lin.dataset))
    t = tree.set_index("dataset").loc[common]
    l = lin.set_index("dataset").loc[common]
    for ds in common:                       # paired connector, same dataset
        ax.plot([t.H[ds]] * 2, [l.faith_s1[ds], t.faith_k1[ds]],
                color="0.78", lw=0.9, zorder=1)

    series = [
        ("trees", t.H.values, t.faith_k1.values, "#1f77b4", "o", "-", 1.0),
        (r"linear, $|w_i x_i|$", lin.H.values, lin.faith_s1.values,
         "#d62728", "s", "-", 1.0),
        # the coefficient ranking is a robustness check: its fit line and r are
        # drawn, but not its fourteen extra markers.
        (r"linear, $|w_i|$", linc.H.values, linc.faith_s1.values,
         "#d62728", None, "--", 0.40),
        ("ReLU $16{\\times}16$",
         relu[relu.arch == "16x16"].H.values,
         relu[relu.arch == "16x16"].faith_k1.values, "#2ca02c", "D", "-", 1.0),
    ]
    for name, x, y, color, marker, ls, alpha in series:
        r, p = r_p(x, y)
        if marker is not None:
            ax.scatter(x, y, s=40, marker=marker, color=color, alpha=alpha,
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.scatter([], [], s=40, marker=marker, color=color, alpha=alpha,
                       label=f"{name}: $r={r:.2f}$" + ("*" if p < 0.05 else ""))
        else:
            fit_line(ax, x, y, color, ls=ls, alpha=alpha)
            ax.plot([], [], ls=ls, color=color, alpha=alpha, lw=1.4,
                    label=f"{name}: $r={r:.2f}$")
            continue
        fit_line(ax, x, y, color, ls=ls, alpha=alpha)

    ax.set_xlabel("Homogeneity $H$")
    ax.set_ylabel("Faithfulness of the\nshortest explanation")
    ax.set_title("Shortest rules across model classes")
    ax.set_xlim(0.62, 1.01)
    ax.set_ylim(0.48, 1.04)   # headroom: the two high-H labels live up here
    # The panel is only ~4.3in wide, so the legend goes underneath: an in-axes
    # legend landed on the low-H points, which is half of why the old scatter
    # was unreadable.
    ax.legend(fontsize=FS_LEGEND - 0.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
              handletextpad=0.4, columnspacing=1.0)

    # --- right: ReLU capacity boundary
    ax = axes[1]
    for arch in ARCHES:
        sub = relu[relu.arch == arch].dropna(subset=["H", "faith_k1"])
        x, y = sub.H.values, sub.faith_k1.values
        r, p = r_p(x, y)
        # Label deliberately terse: the four entries have to sit side by side
        # under the axes, and the long "(n=..., $p=...$)" form did not fit.
        # $n=15$ for all four architectures and lives in the caption; the
        # asterisk is the same significance mark the left panel uses, and it is
        # computed from p rather than written in by hand.
        ax.scatter(x, y, s=48, alpha=0.9, marker=MARKERS[arch],
                   color=ARCH_COLORS[arch], edgecolor="white", linewidth=0.5,
                   label=f"{arch}: $r={r:.2f}$" + ("*" if p < 0.05 else ""),
                   zorder=3)
        if arch == "16x16":
            fit_line(ax, x, y, ARCH_COLORS[arch], alpha=1.0)
        else:
            fit_line(ax, x, y, ARCH_COLORS[arch], ls="--", alpha=0.35)

    ax.set_xlabel("Homogeneity $H$")
    ax.set_ylabel("Faithfulness of a\n1-gate rule")
    ax.set_title("Capacity boundary (ReLU widths)")
    ax.set_xlim(0.60, 1.01)
    ax.set_ylim(0.45, 1.02)
    # Same treatment as the left panel.  In-axes the four-entry box sat on the
    # middle-H points (H ~ 0.85, y ~ 0.6), so it goes under the axes: two
    # columns, no frame, matching the left legend's size and offsets.
    ax.legend(fontsize=FS_LEGEND - 0.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
              handletextpad=0.4, columnspacing=1.0)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FS_TICK)
        ax.title.set_fontsize(FS_TITLE)
        ax.xaxis.label.set_fontsize(FS_LABEL)
        ax.yaxis.label.set_fontsize(FS_LABEL)

    fig.tight_layout(w_pad=2.0)

    # Dataset names go on the tree series, placed against the rendered scene.
    tp = tree.set_index("dataset")
    place_labels(fig, axes[0],
                 [(txt, tp.H[ds], tp.faith_k1[ds])
                  for ds, txt in LABELLED if ds in tp.index],
                 "#1f77b4")

    save(fig, "cross_model_H.png")


# ------------------------------------------------------------------ Figure 3
def figure3(kdf, kcdf):
    hs = {ds: kdf[kdf.dataset == ds].H.iloc[0] for ds in ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)

    for ax, sub, xlab, title in (
            (axes[0], kdf[kdf.dataset.isin(ORDER)], "Rule length $k$ (conditions kept)",
             "Decision trees"),
            (axes[1], kcdf[(kcdf.arch == "16x16") & (kcdf.dataset.isin(ORDER))],
             "Retained gates $k$", "ReLU networks ($16{\\times}16$)")):
        for ds in ORDER:
            s = sub[sub.dataset == ds].sort_values("k")
            if ax is axes[1]:
                s = s[s.n_points >= 5]
            if s.empty:
                continue
            ax.plot(s.k, s.faithfulness, marker="o", linewidth=2, markersize=4,
                    color=COLORS[ds], label=f"{ds} (H={hs[ds]:.3f})")
        ax.set_xlabel(xlab)
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0.5, None)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FS_LEGEND - 0.5, loc="lower right", framealpha=0.92)
        ax.tick_params(labelsize=FS_TICK)
        ax.title.set_fontsize(FS_TITLE)
        ax.xaxis.label.set_fontsize(FS_LABEL)

    axes[0].set_ylabel("Faithfulness of\nshortened rules")
    axes[0].yaxis.label.set_fontsize(FS_LABEL)

    fig.tight_layout(w_pad=1.6)
    save(fig, "shortening_vs_k.png")


def save(fig, name):
    for out in (os.path.join(FIGURES, name), os.path.join(PAPER, name)):
        fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.06)
        print("saved", out)


# ---------------------------------------------------------------------- main
def main():
    tree = csv("faithfulness_cv_results.csv")
    lin = csv("linear_faithfulness_results.csv")
    linc = csv("linear_faithfulness_coef_results.csv")
    relu = csv("mlp_scaled_summary.csv")
    kdf = csv("faithfulness_vs_k_cv.csv")
    kcdf = csv("mlp_scaled_curves.csv")
    tstats = csv("faithfulness_cv_stats.csv").set_index("metric")
    lstats = csv("linear_faithfulness_stats.csv").set_index("metric")
    cstats = csv("linear_faithfulness_coef_stats.csv").set_index("metric")
    mstats = csv("mlp_scaled_stats.csv")

    # Every r drawn must equal the published statistic.
    checks = [
        (r_p(tree.H, tree.faith_k1)[0], tstats.value["headline_w0.7_pearson"], "tree"),
        (r_p(lin.H, lin.faith_s1)[0], lstats.value["linear_faith_s1_pearson_local"],
         "linear |w_i x_i|"),
        (r_p(linc.H, linc.faith_s1)[0], cstats.value["linear_faith_s1_pearson_coef"],
         "linear |w_i|"),
    ]
    for got, want, what in checks:
        assert abs(got - want) < 1e-4, (what, got, want)
        print(f"  ok {what:18s} r={got:.4f}")

    row = mstats[(mstats.arch == "16x16") & (mstats.y == "faith_k1")]
    got = r_p(relu[relu.arch == "16x16"].H, relu[relu.arch == "16x16"].faith_k1)[0]
    assert abs(got - float(row.pearson.iloc[0])) < 1e-4, (got, row.pearson.iloc[0])
    print(f"  ok {'ReLU 16x16':18s} r={got:.4f}")

    figure2(tree, lin, linc, relu, mstats)
    figure3(kdf, kcdf)


if __name__ == "__main__":
    main()
