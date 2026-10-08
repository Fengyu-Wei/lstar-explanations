"""Post-review audit of complete projections; outputs never replace old results.

Frozen seed range, data splits, preprocessing and model settings are inherited
from crossmodel_replication.py. Boundary convention is native sklearn (>0).
"""
import json
import time
import warnings
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from utility_experiment import ROOT, datasets, split_data, TreeRules, mask
from crossmodel_replication import forward

OUT = ROOT / 'results/exact-projection-audit/results'

def frozen_affine(x, model):
    """Explicit coefficient matrices, used to test the frozen construction."""
    gates, _, _ = forward(x[None, :], model)
    A = np.eye(len(x), dtype=np.float64)
    b = np.zeros(len(x), dtype=np.float64)
    constraints = []
    for z, W, bias in zip(gates, model.coefs_[:-1], model.intercepts_[:-1]):
        A, b = A @ W, b @ W + bias
        active = z[0] > 0
        constraints.append((A.copy(), b.copy(), active))
        A = A * active
        b = b * active
    return constraints, A @ model.coefs_[-1], b @ model.coefs_[-1] + model.intercepts_[-1]

def frozen_mask(Z, constraints, a, b, c):
    inside = np.ones(len(Z), dtype=bool)
    for A, bias, active in constraints:
        inside &= ((Z.astype(np.float64) @ A + bias > 0) == active).all(axis=1)
    score = (Z.astype(np.float64) @ a + b).ravel()
    return inside & ((score > 0) == c), inside, score

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, fits = [], []
    t0 = time.time()
    with threadpool_limits(limits=1):
        for name, X, y, names in datasets():
            for seed in range(5):
                (Xf, Xs, Xr, Xq), indices, _ = split_data(X, y, seed)
                fi, si, ri, qi = indices
                scaler = StandardScaler().fit(Xf)
                Xf, Xs, Xr, Xq = [scaler.transform(Z).astype(np.float32) for Z in [Xf, Xs, Xr, Xq]]
                models = {'tree': DecisionTreeClassifier(max_depth=8, random_state=seed),
                          'linear': LogisticRegression(C=1, max_iter=2000, random_state=seed)}
                if name != 'internet-ads':
                    models['relu'] = MLPClassifier(hidden_layer_sizes=(16,16), max_iter=1500, tol=1e-5, random_state=seed)
                for kind, model in models.items():
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always')
                        model.fit(Xf, y[fi])
                    p, pq = model.predict(Xr), model.predict(Xq)
                    fits.append({'dataset': name, 'seed': seed, 'model': kind,
                                 'accuracy': float(np.mean(p == y[ri])),
                                 'model_majority': float(np.bincount(p, minlength=2).max()/len(p)),
                                 'warnings': json.dumps([str(w.message) for w in ws])})
                    if kind == 'tree':
                        tr = TreeRules(model, Xs)
                    if kind == 'relu':
                        _, sr, logr = forward(Xr, model)
                        _, sq, _ = forward(Xq, model)
                        assert np.array_equal((logr.ravel()>0).astype(int), p)
                    for j, x in enumerate(Xq):
                        c = int(pq[j]); st = time.perf_counter()
                        numerical_disagreement = 0; score_error = 0.0
                        gate_support = np.nan; gate_success = np.nan
                        if kind == 'tree':
                            path, _, _ = tr.path(x)
                            mr = mask(path, Xr); contains = bool(mask(path, x[None,:])[0])
                            length = len(path)
                        elif kind == 'linear':
                            score = model.decision_function(Xr)
                            assert np.array_equal((score>0).astype(int), p)
                            mr = (score>0)==c
                            contains = bool((model.decision_function(x[None,:])[0]>0)==c)
                            length = 1
                        else:
                            constraints, a, b = frozen_affine(x, model)
                            mr, cell, frozen_score = frozen_mask(Xr, constraints, a, b, c)
                            mq, _, _ = frozen_mask(x[None,:], constraints, a, b, c)
                            contains = bool(mq[0]); length = sum(len(z[2]) for z in constraints)+1
                            native_cell = (sr == sq[j]).all(axis=1)
                            native_rule = native_cell & ((logr.ravel()>0)==c)
                            numerical_disagreement = int(np.sum(mr != native_rule))
                            score_error = float(np.max(np.abs(frozen_score[cell]-logr.ravel()[cell]))) if cell.any() else 0.0
                            gate_support = int(native_cell.sum()); gate_success = int(np.sum(p[native_cell]==c))
                        n = int(mr.sum()); success = int(np.sum(p[mr]==c))
                        assert contains, (name, seed, kind, j, 'query membership')
                        assert success == n, (name, seed, kind, j, 'model disagreement')
                        rows.append({'dataset': name, 'seed': seed, 'model': kind, 'query_index': int(qi[j]),
                                     'class': c, 'constraints': length, 'contains_query': int(contains),
                                     'ref_count': n, 'ref_success': success, 'precision': success/n if n else np.nan,
                                     'coverage': n/len(Xr), 'evaluable': int(n>0),
                                     'projection_ms': (time.perf_counter()-st)*1000,
                                     'numerical_mask_disagreements': numerical_disagreement,
                                     'frozen_score_max_error': score_error,
                                     'gate_count': gate_support, 'gate_success': gate_success})
            pd.DataFrame(rows).to_csv(OUT/'queries.csv', index=False)
            pd.DataFrame(fits).to_csv(OUT/'fits.csv', index=False)
            print(f'{name}: {len(fits)} fits, {len(rows)} projections, {time.time()-t0:.1f}s', flush=True)
    data = pd.DataFrame(rows)
    summary = data.groupby(['dataset','seed','model'])[['constraints','precision','coverage','evaluable','projection_ms']].mean().groupby('model').mean()
    summary.to_csv(OUT/'summary.csv')
    diagnostics = {'fits': len(fits), 'projections': len(rows), 'seconds': time.time()-t0,
                   'query_membership_failures': int((data.contains_query==0).sum()),
                   'model_disagreements': int((data.ref_count-data.ref_success).sum()),
                   'numerical_mask_disagreements': int(data.numerical_mask_disagreements.sum()),
                   'max_frozen_score_error': float(data.frozen_score_max_error.max()),
                   'platform': platform.platform(), 'python': platform.python_version(),
                   'versions': {k: importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']}}
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    print(summary.to_string()); print(json.dumps(diagnostics), flush=True)

if __name__ == '__main__':
    run()
