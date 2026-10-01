# Faithful Explanations — reproducibility artifact

Code, result tables and figures for

> **A Unified Logical Framework for Faithful Explanations: Translation and Projection of Linear and Tree Models**
> F. Wei, D. Zhang — submitted to *Knowledge-Based Systems*

This repository holds exactly what is needed to reproduce the paper's experimental section: the five scripts that produce its numbers and figures, the ten result tables they are read from, and the two result figures themselves. Every experimental number quoted in the paper is taken from `results/`.

## Layout

```
code/             analysis scripts (Python)
results/          archived CSV tables — the reference outputs
figures/          the two result figures (PNG, 150 dpi)
requirements.txt  pinned environment
```

## Environment

Python 3.14.

```
pip install -r requirements.txt
```

The recorded runs used numpy 2.5.2, pandas 3.0.3, scikit-learn 1.9.0, matplotlib 3.11.0 and scipy 1.18.0.

## Datasets

Fourteen binary datasets are used in the tree and linear studies. The ReLU study uses thirteen of them — the same set without the 1558-dimensional `climate` snapshot — plus `heart` and `vehicle`, for fifteen in total. The `n`, `d` and `H` columns below are read from `results/faithfulness_cv_results.csv` and `results/mlp_scaled_summary.csv`.

| dataset | source | identifier | n | d | H |
| --- | --- | --- | --- | --- | --- |
| iris | scikit-learn `load_iris` | — | 100 | 4 | 0.927 |
| wine | scikit-learn `load_wine` | — | 130 | 13 | 0.943 |
| breast_cancer | scikit-learn `load_breast_cancer` | — | 569 | 30 | 0.916 |
| digits-35 | scikit-learn `load_digits`, classes {3, 5} | — | 365 | 64 | 0.979 |
| synthetic_d8 | `make_classification`, `random_state=42` | — | 500 | 8 | 0.877 |
| synthetic_d20 | `make_classification`, `random_state=1` | — | 800 | 20 | 0.777 |
| synthetic_d50 | `make_classification`, `random_state=2` | — | 1000 | 50 | 0.770 |
| synthetic_lowsep | `make_classification`, `random_state=3` | — | 800 | 10 | 0.755 |
| banknote | UCI *banknote-authentication* | OpenML `data_id=1462` | 1372 | 4 | 0.933 |
| blood | UCI *blood-transfusion-service-center* | OpenML `data_id=1464` | 748 | 4 | 0.731 |
| diabetes | UCI *Pima Indians diabetes* | OpenML `data_id=37` | 768 | 8 | 0.667 |
| ionosphere | UCI *ionosphere* | OpenML `data_id=59` | 351 | 34 | 0.850 |
| sonar | UCI *sonar* | OpenML `data_id=40` | 208 | 60 | 0.749 |
| climate | UCI *climate-model-simulation-crashes* | OpenML `data_id=40978` | 3279 | 1558 | 0.953 |
| heart | UCI *heart-statlog* | OpenML `data_id=53` | 270 | 13 | 0.766 |
| vehicle | UCI *vehicle silhouettes* | OpenML `data_id=54` | 435 | 18 | 0.872 |

The identifiers are declared in `_OPENML_IDS` ([`code/faithfulness_cv_experiment.py`](code/faithfulness_cv_experiment.py)) and `EXTRA_OPENML` ([`code/mlp_gatepath_experiment.py`](code/mlp_gatepath_experiment.py)). They are fetched through scikit-learn, so the OpenML copy is what the runs saw; the UCI originals are named for provenance only.

**Preprocessing.** Every dataset passes through one shared path, `to_binary()`:

1. non-finite entries are replaced by zero — some OpenML tables carry missing values;
2. the target is reduced to its two most frequent classes, so every dataset is binary;
3. features are standardized with `StandardScaler`.

Two dataset-specific notes:

- `digits-35` is `load_digits` binarized to classes {3, 5} (183 + 182 = 365 samples). The literal {0, 1} pair holds only 360, so the two most frequent classes are used instead — uniformly for every dataset, not as a special case.
- OpenML `data_id=40978` fetches a wider repeated-measure table: after preprocessing it yields the n = 3279, d = 1558 snapshot recorded in the tables. The "(2 classes, 20 features)" text shipped with the raw OpenML description no longer matches the fetched layout, which is why the identifier — not that description — is recorded here.

The four synthetic sets are generated in-process with fixed `random_state` values, so they need no download.

## Protocol

All three studies share one protocol.

- **Seed.** `np.random.seed(42)` at module level, and `random_state=42` on every estimator (`KMeans`, `StratifiedKFold`, `DecisionTreeClassifier`, `LogisticRegression`, `MLPClassifier`). Confidence intervals use 2000 bootstrap resamples with seed 42.
- **Cross-validation.** `StratifiedKFold(shuffle=True, random_state=42)` with `n_splits = max(2, min(5, smallest class count))` — five folds unless a class is too small to support them.
- **Homogeneity.** `H = 0.7 * kNN + 0.3 * cluster`, where kNN consistency is the mean agreement of a point with its 5 nearest neighbours and cluster is the mean KMeans purity over 5 clusters.
- **Faithfulness.** For a test input `x`, model `M` and pool `P` of real points, the first `k` conditions of `x`'s rule define a region; `faithful_k(x)` is the fraction of that region on which the full model agrees with its prediction at `x`:

  ```
  faithful_k(x) = |{x' in P : x' satisfies the first k conditions, M(x') = M(x)}|
                  --------------------------------------------------------------
                          |{x' in P : x' satisfies the first k conditions}|
  ```

  `P` is the **training fold**, never uniformly sampled boxes: the off-manifold points such a box produces in high dimension were the flaw of the earlier single-split prototype.
- **Study settings.** The tree study evaluates depths 2–8, samples up to 30 test points per fold, and requires a minimum region cover of 5 points. The linear study ranks features locally by `|w_i x_i|` and, as a robustness check, globally by `|w_i|`. The ReLU study ranks gates by `|d(logit)/d z_gate|` at `x`.

## Scripts

| script | study | writes |
| --- | --- | --- |
| [`code/faithfulness_cv_experiment.py`](code/faithfulness_cv_experiment.py) | **trees** — 14 datasets, depths 2–8 | `faithfulness_cv_results.csv`, `faithfulness_cv_stats.csv`, `faithfulness_vs_k_cv.csv` |
| [`code/linear_faithfulness_experiment.py`](code/linear_faithfulness_experiment.py) | **linear** — logistic regression on the same 14 datasets | `linear_faithfulness_results.csv`, `linear_faithfulness_coef_results.csv`, `linear_faithfulness_stats.csv`, `linear_faithfulness_coef_stats.csv` |
| [`code/mlp_gatepath_experiment.py`](code/mlp_gatepath_experiment.py) | **ReLU** — architectures 16x16, 32x16, 128x64, 128x64x32 on 15 datasets | `mlp_scaled_summary.csv`, `mlp_scaled_curves.csv`, `mlp_scaled_stats.csv` |
| [`code/unified_figures.py`](code/unified_figures.py) | the paper's two result figures, drawing on all three studies | `cross_model_H.png`, `shortening_vs_k.png` |
| [`code/xai_framework.py`](code/xai_framework.py) | the `L*` formula algebra, the tree and linear translators, and the axiom tests | — (imported by the scripts above) |

The three experiment scripts also save the single-study PNGs they plot; only the two merged figures are used in the paper, and only those two are archived here.

`unified_figures.py` reads only the cached CSV tables — no network is refitted — and asserts each correlation it draws against the corresponding stats CSV, so the figure cannot drift away from the numbers in the text.

## Reproducing

```
pip install -r requirements.txt

cd results
python ../code/faithfulness_cv_experiment.py
python ../code/linear_faithfulness_experiment.py
python ../code/mlp_gatepath_experiment.py
cd ..
python code/unified_figures.py
```

The three experiment scripts write their CSV and PNG outputs to the **current working directory**, which is why they are run from `results/`; the archived tables in `results/` are exactly what they produce there. `unified_figures.py` is the exception — it resolves every path from its own location, so it can be run from anywhere.

**One caveat when running the figure script:** its `save()` writes each figure twice — into `figures/` and into a `../paper/KBS/` tree that belongs to the manuscript, which is not part of this repository. That second write raises `FileNotFoundError` unless you create the directory or drop that path from `save()`. The figures already in `figures/` are the released copies and are unaffected.

The OpenML datasets download on first use into scikit-learn's cache; `climate` is by far the largest (d = 1558).

## Citing

If you use this artifact, please cite the paper above. The release is tagged `v1.0.0`.
