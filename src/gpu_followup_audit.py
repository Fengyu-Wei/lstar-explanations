"""Post-review GPU audit; never overwrites the original experimental results.

Run in the GPU environment described in README.md. CPU reconstructs the original
CART algorithm and prepares tables; CUDA evaluates regions and resamples means.
Neural controls reuse recorded predictions/supports from identical audited fits.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/reviewer-followup-gpu/results'
OUT.mkdir(parents=True, exist_ok=True)
TMP = ROOT / '.cache/cuda-temp'
TMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TMP)
os.environ.setdefault('CUPY_CACHE_DIR', str(ROOT / '.cache/cupy'))
# This optional script loads the CUDA backend after configuring local caches.
import cupy as cp
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
import utility_experiment as utility

OLD = ROOT / 'experiments/heldout-utility/results'
KEY = ['dataset', 'seed', 'depth', 'query_index']
B = 5000
RNG_SEED = 20261007
VALIDATION = {}


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def bootstrap(frame, metrics, name, query_average=False):
    """Dataset/seed hierarchical bootstrap on GPU, with CPU numerical checks."""
    if query_average:
        frame = frame.groupby(KEY, as_index=False)[metrics].mean()
    means = frame.groupby(['dataset', 'seed'])[metrics].mean()
    datasets = sorted(frame.dataset.unique())
    seeds = sorted(frame.seed.unique())
    vals = means.reindex(pd.MultiIndex.from_product([datasets, seeds])).to_numpy()
    vals = vals.reshape(len(datasets), len(seeds), len(metrics))
    rng = np.random.default_rng(RNG_SEED)
    di = rng.integers(0, len(datasets), (B, len(datasets)))
    si = rng.integers(0, len(seeds), (B, len(datasets), len(seeds)))
    gpu = cp.asarray(vals, dtype=cp.float64)
    sampled = gpu[cp.asarray(di)[:, :, None], cp.asarray(si)]
    with np.errstate(invalid='ignore'):
        estimates = cp.asnumpy(cp.nanmean(cp.nanmean(gpu, axis=1), axis=0))
        boots = cp.asnumpy(cp.nanmean(cp.nanmean(sampled, axis=2), axis=1))
    cpu_estimate = np.nanmean(np.nanmean(vals, axis=1), axis=0)
    cpu_boots = np.nanmean(np.nanmean(vals[di[:, :, None], si], axis=2), axis=1)
    assert np.allclose(estimates, cpu_estimate, atol=1e-12, equal_nan=True)
    assert np.allclose(boots, cpu_boots, atol=1e-12, equal_nan=True)
    rows = []
    for j, metric in enumerate(metrics):
        lo, hi = np.nanquantile(boots[:, j], [.025, .975])
        rows.append({'metric': metric, 'mean': estimates[j], 'ci_lo': lo,
                     'ci_hi': hi, 'tasks_with_data': int(np.isfinite(vals[:, :, j]).any(axis=1).sum()),
                     'task_seed_cells': int(np.isfinite(vals[:, :, j]).sum())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / (name + '_ci.csv'), index=False)
    means.to_csv(OUT / (name + '_by_task_seed.csv'))
    VALIDATION[name + '_gpu_cpu_bootstrap_max_difference'] = float(np.nanmax(np.abs(boots-cpu_boots)))
    return result


def tree_predict_gpu(X, model):
    t = model.tree_
    left = cp.asarray(t.children_left)
    right = cp.asarray(t.children_right)
    feature = cp.asarray(np.maximum(t.feature, 0))
    thresholds = cp.asarray(t.threshold)
    labels = cp.asarray(t.value.argmax(axis=2).ravel())
    values = cp.asarray(X, dtype=cp.float64)
    nodes = cp.zeros(len(X), dtype=cp.int64)
    ids = cp.arange(len(X))
    for _ in range(t.max_depth):
        child = cp.where(values[ids, feature[nodes]] <= thresholds[nodes], left[nodes], right[nodes])
        nodes = cp.where(left[nodes] < 0, nodes, child)
    predictions = labels[nodes]
    assert np.array_equal(cp.asnumpy(predictions), model.predict(X))
    return values, predictions


def prefix_counts_gpu(X, predictions, paths, classes, max_depth):
    n_queries = len(paths)
    features = np.zeros((n_queries, max_depth), dtype=int)
    thresholds = np.zeros((n_queries, max_depth), dtype=float)
    directions = np.zeros((n_queries, max_depth), dtype=bool)
    valid = np.zeros((n_queries, max_depth), dtype=bool)
    for j, path in enumerate(paths):
        for k, (f, op, threshold) in enumerate(path):
            features[j, k] = f
            thresholds[j, k] = threshold
            directions[j, k] = op == '<='
            valid[j, k] = True
    values = X[:, cp.asarray(features)].transpose(1, 0, 2)
    tests = cp.where(cp.asarray(directions)[:, None, :],
                     values <= cp.asarray(thresholds)[:, None, :],
                     values > cp.asarray(thresholds)[:, None, :])
    tests |= ~cp.asarray(valid)[:, None, :]
    regions = cp.concatenate([cp.ones((n_queries, len(X), 1), dtype=cp.bool_),
                              cp.cumprod(tests, axis=2).astype(cp.bool_)], axis=2)
    n = regions.sum(axis=1)
    agree = predictions[None, :, None] == cp.asarray(classes)[:, None, None]
    success = (regions & agree).sum(axis=1)
    return n, success


def tree_audit():
    curves = pd.read_csv(OLD / 'paired_curves.csv').set_index(KEY + ['k'])
    recorded = pd.read_csv(OLD / 'queries.csv')
    exact = recorded[recorded.method == 'full_path'].set_index(KEY)
    cases_policy = json.loads((OLD / 'cases.json').read_text())
    case_keys = {(r['dataset'], r['seed'], r['depth'], r['query_index']) for r in cases_policy}
    recorded_manifest = {z['dataset']: z for z in json.loads((ROOT/'data/manifest.json').read_text())}
    # The imported loader may write its manifest. Redirect that write to this audit.
    utility.ROOT = OUT
    (OUT/'data').mkdir(exist_ok=True)
    prefixes, siblings, localized, selections, cases = [], [], [], [], []
    fits = []
    started = time.perf_counter()
    checks = 0
    max_identity_error = 0.0
    with threadpool_limits(limits=1):
        for name, X, y, names in utility.datasets():
            digest = hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
            assert digest == recorded_manifest[name]['sha256'], (name, 'data identity')
            for seed in range(10):
                (Xf, Xc, Xr, Xq), indices, _ = utility.split_data(X, y, seed)
                fi, _, _, qi = indices
                for depth in [4, 8]:
                    fit_started = time.perf_counter()
                    model = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(Xf, y[fi])
                    reconstruction_seconds = time.perf_counter()-fit_started
                    tr = utility.TreeRules(model, Xc)
                    details = [tr.path(x) for x in Xq]
                    paths = [a[0] for a in details]
                    classes = np.array([a[2] for a in details])
                    vg_c, pg_c = tree_predict_gpu(Xc, model)
                    vg_r, pg_r = tree_predict_gpu(Xr, model)
                    cn, cs = prefix_counts_gpu(vg_c, pg_c, paths, classes, depth)
                    rn, rs = prefix_counts_gpu(vg_r, pg_r, paths, classes, depth)
                    # Main numerical budget calculations and sibling losses run on CUDA.
                    lengths = np.array([len(p) for p in paths])
                    leaf_n = rn[cp.arange(len(paths)), cp.asarray(lengths)]
                    q = cp.where(rn > 0, leaf_n[:, None]/cp.maximum(rn, 1), cp.nan)
                    precision = cp.where(rn > 0, rs/cp.maximum(rn, 1), cp.nan)
                    bad = rn-rs
                    sibling_n = rn[:, :-1]-rn[:, 1:]
                    sibling_bad = bad[:, :-1]-bad[:, 1:]
                    assert bool(cp.all(sibling_n >= 0)) and bool(cp.all(sibling_bad >= 0))
                    cal_q = cp.where(cn > 0, cn[cp.arange(len(paths)), cp.asarray(lengths)][:, None]/cp.maximum(cn,1), cp.nan)
                    cn, cs, rn, rs, q, precision, bad, sibling_n, sibling_bad, cal_q = [
                        cp.asnumpy(z) for z in [cn, cs, rn, rs, q, precision, bad, sibling_n, sibling_bad, cal_q]]
                    fits.append({'dataset':name, 'seed':seed, 'depth':depth,
                                 'cpu_reconstruction_seconds':reconstruction_seconds,
                                 'nodes':model.tree_.node_count, 'queries':len(paths)})
                    for j, (path, nodes, c) in enumerate(details):
                        common = {'dataset':name, 'seed':seed, 'depth':depth,
                                  'query_index':int(qi[j]), 'class':int(c), 'full_length':len(path)}
                        old = exact.loc[(name,seed,depth,int(qi[j]))]
                        assert old['class'] == c and old.full_length == len(path)
                        assert old.ref_count == rn[j,len(path)] and old.ref_success == rs[j,len(path)]
                        for k in range(1,9):
                            original = curves.loc[(name,seed,depth,int(qi[j]),k)]
                            kk = min(k,len(path))
                            assert original.ref_count == rn[j,kk] and original.ref_success == rs[j,kk]
                            checks += 1
                        for k in range(len(path)+1):
                            pure = tr.pure[nodes[k]] == {int(c)}
                            prefixes.append(common | {'k':k, 'cal_count':int(cn[j,k]),
                                'cal_success':int(cs[j,k]), 'cal_q':float(cal_q[j,k]),
                                'ref_count':int(rn[j,k]), 'ref_success':int(rs[j,k]),
                                'q':float(q[j,k]), 'precision':float(precision[j,k]),
                                'gap':float(precision[j,k]-q[j,k]), 'structural':int(pure),
                                'budget_pass':float(q[j,k] >= .95) if rn[j,k] else np.nan,
                                'structural_budget_miss':float(q[j,k] < .95) if pure and rn[j,k] else np.nan})
                            ref_bad = bad[j,k]
                            if rn[j,k]:
                                err = abs(sibling_bad[j,k:len(path)].sum()/rn[j,k]-(1-precision[j,k]))
                                max_identity_error = max(max_identity_error,float(err))
                                assert err < 1e-12 and precision[j,k]+1e-12 >= q[j,k]
                            for s in range(k,len(path)):
                                cal_n = int(cn[j,s]-cn[j,s+1])
                                cal_bad = int((cn[j,s]-cs[j,s])-(cn[j,s+1]-cs[j,s+1]))
                                siblings.append(common | {'prefix_k':k,'split_j':s,
                                    'cal_sibling_count':cal_n,'cal_sibling_bad':cal_bad,
                                    'ref_sibling_count':int(sibling_n[j,s]),
                                    'ref_sibling_bad':int(sibling_bad[j,s]),
                                    'ref_deficit_contribution':float(sibling_bad[j,s]/rn[j,k]) if rn[j,k] else np.nan})
                            if 1 <= k < len(path):
                                cal_ns = cn[j,k:len(path)]-cn[j,k+1:len(path)+1]
                                cal_bs = (cn[j,k:len(path)]-cs[j,k:len(path)])-(cn[j,k+1:len(path)+1]-cs[j,k+1:len(path)+1])
                                eligible = np.flatnonzero(cal_ns >= 5)
                                if len(eligible)>=2 and cal_bs[eligible].max()>0:
                                    chosen = int(eligible[np.argmax(cal_bs[eligible])])+k
                                    share = sibling_bad[j,chosen]/ref_bad if ref_bad else np.nan
                                    uniform = sibling_bad[j,k+eligible].mean()/ref_bad if ref_bad else np.nan
                                    hit = float(sibling_bad[j,chosen] == sibling_bad[j,k:len(path)].max()) if ref_bad else np.nan
                                    localized.append(common | {'k':k,'chosen_split':chosen,
                                        'eligible_siblings':len(eligible),'ref_error_present':int(ref_bad>0),
                                        'selected_ref_supported':int(sibling_n[j,chosen]>0),
                                        'selected_error_share':float(share),'uniform_error_share':float(uniform),
                                        'excess_share':float(share-uniform),'top_error_hit':float(hit)})
                        passed = np.flatnonzero((cn[j,:len(path)+1]>=5)&(cal_q[j,:len(path)+1]>=.95))
                        chosen_k = int(passed[0]) if len(passed) else len(path)
                        selections.append(common | {'method':'mass_budget_prefix','length':chosen_k,
                            'shortened':int(chosen_k<len(path)), 'ref_count':int(rn[j,chosen_k]),
                            'ref_success':int(rs[j,chosen_k]), 'coverage':float(rn[j,chosen_k]/len(Xr)),
                            'precision':float(precision[j,chosen_k]), 'evaluable':int(rn[j,chosen_k]>0),
                            'below_target':float(precision[j,chosen_k]<.95) if rn[j,chosen_k] else np.nan,
                            'structural':int(tr.pure[nodes[chosen_k]]=={int(c)})})
                        if (name,seed,depth,int(qi[j])) in case_keys:
                            cases.append(common | {'predicates':[[names[f],op,float(v)] for f,op,v in path],
                                'prefixes':[{'k':k,'support':int(rn[j,k]),'success':int(rs[j,k]),
                                  'q':float(q[j,k]) if rn[j,k] else None,
                                  'precision':float(precision[j,k]) if rn[j,k] else None,
                                  'sibling_bad':[int(v) for v in sibling_bad[j,k:len(path)]]}
                                  for k in range(len(path)+1)]})
            print(f'Tree audit {name}: {len(fits)} fits, {time.perf_counter()-started:.1f}s',flush=True)
    p=pd.DataFrame(prefixes); s=pd.DataFrame(siblings); loc=pd.DataFrame(localized)
    new=pd.DataFrame(selections)
    p.to_csv(OUT/'prefixes.csv',index=False); s.to_csv(OUT/'siblings.csv',index=False)
    loc.to_csv(OUT/'localization.csv',index=False); new.to_csv(OUT/'mass_budget_queries.csv',index=False)
    pd.DataFrame(fits).to_csv(OUT/'tree_fits.csv',index=False); save_json('case_paths.json',cases)
    proper = p[(p.k>=1)&(p.k<p.full_length)&(p.ref_count>0)].copy()
    proper['zero_leaf_support'] = proper.q.eq(0).astype(float)
    bootstrap(proper,['q','precision','gap','budget_pass','zero_leaf_support','structural_budget_miss'],
              'budget_diagnostics',query_average=True)
    assert len(loc)>0
    bootstrap(loc,['selected_error_share','uniform_error_share','excess_share','top_error_hit',
                   'ref_error_present','selected_ref_supported'],'sibling_localization',query_average=True)
    controls = pd.concat([new,recorded[recorded.method.isin(['full_path','pure_prefix','empirical_prefix'])]],ignore_index=True)
    results=[]
    for method,z in controls.groupby('method'):
        summary=bootstrap(z,['length','precision','coverage','evaluable','below_target','shortened'],method)
        summary['method']=method; results.append(summary)
    pd.concat(results).to_csv(OUT/'budget_controls_ci.csv',index=False)
    paired=new.merge(recorded[recorded.method=='empirical_prefix'],on=KEY,suffixes=('_budget','_empirical'),validate='one_to_one')
    assert (paired.length_budget>=paired.length_empirical).all()
    paired['length_delta']=paired.length_budget-paired.length_empirical
    paired['coverage_delta']=paired.coverage_budget-paired.coverage_empirical
    bootstrap(paired,['length_delta','coverage_delta'],'budget_vs_empirical')
    VALIDATION.update({'reconstructed_tree_fits':len(fits),'tree_queries':len(new),
        'stored_curve_counts_reproduced':checks,'sibling_identity_max_error':max_identity_error,
        'proper_supported_prefix_records':len(proper),'proper_prefix_queries':int(proper.groupby(KEY).ngroups),
        'localization_candidate_records':len(loc),'localization_queries':int(loc.groupby(KEY).ngroups),
        'gpu_tree_prediction_cpu_disagreements':0})


def neural_controls():
    full=pd.read_csv(ROOT/'experiments/exact-projection-audit/results/queries.csv')
    full=full[full.model=='relu']
    partial=pd.read_csv(ROOT/'experiments/crossmodel-robustness/results/queries.csv')
    partial=partial[partial.model=='relu']
    keys=['dataset','seed','query_index']
    base=partial[partial.k==32].merge(full,on=keys,suffixes=('_gate','_full'),validate='one_to_one')
    assert (base['class_gate']==base['class_full']).all()
    assert (base.ref_count_full<=base.ref_count_gate).all()
    assert (base.ref_count_gate==base.gate_count).all()
    sizes=pd.read_csv(OLD/'fits.csv'); sizes=sizes[sizes.depth==8][['dataset','seed','n_ref']]
    base=base.merge(sizes,on=['dataset','seed'],validate='many_to_one')
    gn=cp.asarray(base.constant_precision.to_numpy())*cp.asarray(base.n_ref.to_numpy())
    rounded=cp.rint(gn).astype(cp.int64)
    assert float(cp.max(cp.abs(gn-rounded)))<1e-9
    dc_count=cp.asnumpy(rounded)
    assert (base.ref_count_full.to_numpy()<=dc_count).all()
    all_rows=[]
    for region,counts,success in [('class_region',dc_count,dc_count),
                                  ('complete_cell',base.ref_count_full,base.ref_success),
                                  ('all_gates',base.ref_count_gate,base.gate_success)]:
        n=cp.asarray(np.asarray(counts)); good=cp.asarray(np.asarray(success)); nr=cp.asarray(base.n_ref.to_numpy())
        stats=cp.asnumpy(cp.stack([good/cp.maximum(n,1),n/nr,(n>0).astype(cp.float64)],axis=1))
        for j,(_,z) in enumerate(base.iterrows()):
            all_rows.append({k:z[k] for k in keys}|{'region':region,'class':int(z.class_full),
                'ref_count':int(n[j]),'ref_success':int(good[j]),'precision':stats[j,0] if n[j] else np.nan,
                'coverage':stats[j,1],'evaluable':stats[j,2]})
    for _,z in partial[partial.k.isin([1,2,4,8])].iterrows():
        all_rows.append({k:z[k] for k in keys}|{'region':f'gates_{int(z.k)}','class':int(z['class']),
            'ref_count':int(z.ref_count),'ref_success':int(round(z.precision*z.ref_count)) if z.ref_count else 0,
            'precision':z.precision,'coverage':z.coverage,'evaluable':z.evaluable})
    d=pd.DataFrame(all_rows); d.to_csv(OUT/'neural_regions.csv',index=False)
    summaries=[]
    for region,z in d.groupby('region'):
        r=bootstrap(z,['precision','coverage','evaluable'],'neural_'+region); r['region']=region; summaries.append(r)
    pd.concat(summaries).to_csv(OUT/'neural_regions_ci.csv',index=False)
    paired=base[keys].copy()
    paired['class_minus_cell_coverage']=base.constant_precision-base.coverage_full
    paired['class_minus_cell_evaluable']=(dc_count>0).astype(float)-base.evaluable_full
    bootstrap(paired,['class_minus_cell_coverage','class_minus_cell_evaluable'],'neural_representation_delta')
    VALIDATION.update({'same_network_fits':int(base.groupby(['dataset','seed']).ngroups),
                       'same_network_queries':len(base),'neural_support_inclusion_violations':0,
                       'class_support_recovery_max_error':float(cp.max(cp.abs(gn-rounded)))})


def matched_uncertainty():
    greedy=pd.read_csv(OLD/'queries.csv'); greedy=greedy[greedy.method=='empirical_delete']
    minimum=pd.read_csv(ROOT/'experiments/matched-subset-audit/results/queries.csv')
    minimum=minimum[minimum.method=='minimum_empirical_subset']
    pairs=minimum.merge(greedy,on=KEY,suffixes=('_minimum','_greedy'),validate='one_to_one')
    assert len(pairs)==105
    pairs['both_supported']=((pairs.ref_count_minimum>0)&(pairs.ref_count_greedy>0)).astype(float)
    a=cp.asarray(pairs.precision_minimum.to_numpy()); b=cp.asarray(pairs.precision_greedy.to_numpy())
    both=cp.isfinite(a)&cp.isfinite(b)
    pairs['miss_delta']=cp.asnumpy(cp.where(both,(a<.95).astype(cp.float64)-(b<.95).astype(cp.float64),cp.nan))
    pairs['shortfall_delta']=cp.asnumpy(cp.where(both,cp.maximum(0,.95-a)-cp.maximum(0,.95-b),cp.nan))
    pairs['minimum_shortfall']=cp.asnumpy(cp.where(both,cp.maximum(0,.95-a),cp.nan))
    pairs['greedy_shortfall']=cp.asnumpy(cp.where(both,cp.maximum(0,.95-b),cp.nan))
    pairs.to_csv(OUT/'matched_pairs.csv',index=False)
    metrics=['miss_delta','shortfall_delta','minimum_shortfall','greedy_shortfall','both_supported']
    bootstrap(pairs,metrics,'matched_reliability')
    loo=[]
    for dataset in sorted(pairs.dataset.unique()):
        r=bootstrap(pairs[pairs.dataset!=dataset],['miss_delta','shortfall_delta'],'loo_'+dataset)
        r['excluded_task']=dataset; loo.append(r)
    pd.concat(loo).to_csv(OUT/'matched_leave_one_out.csv',index=False)
    VALIDATION['matched_queries']=len(pairs)
    VALIDATION['matched_both_supported']=int(pairs.both_supported.sum())


def main():
    inputs=[OLD/'paired_curves.csv',OLD/'queries.csv',OLD/'fits.csv',OLD/'cases.json',
            ROOT/'data/manifest.json', ROOT/'data/splits.json',
            ROOT/'experiments/crossmodel-robustness/results/queries.csv',
            ROOT/'experiments/exact-projection-audit/results/queries.csv',
            ROOT/'experiments/matched-subset-audit/results/queries.csv']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    assert cp.cuda.runtime.getDeviceCount()>0, 'GPU required; no silent CPU fallback'
    # Trigger a real CUDA computation before any experiment, failing early if unusable.
    assert float((cp.arange(10000,dtype=cp.float64)**2).sum())==333283335000.0
    prop=cp.cuda.runtime.getDeviceProperties(0)
    started=time.perf_counter()
    tree_audit(); neural_controls(); matched_uncertainty()
    cp.cuda.Stream.null.synchronize()
    after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    assert hashes==after, 'Original evidence changed'
    save_json('input_hashes.json',hashes)
    save_json('validation.json',VALIDATION|{'original_inputs_unchanged':True})
    versions={p:importlib.metadata.version(p) for p in ['cupy-cuda12x','numpy','pandas','scipy','scikit-learn']}
    save_json('runtime.json',{'seconds':time.perf_counter()-started,'platform':platform.platform(),
        'python':sys.version,'versions':versions,'device':prop['name'].decode(),
        'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(),'cuda_driver':cp.cuda.runtime.driverGetVersion(),
        'bootstrap_resamples':B,'rng_seed':RNG_SEED,
        'execution':'CUDA float64 region evaluation and bootstrap; original CART reconstructed on CPU; neural controls reuse fixed-fit records'})
    print(json.dumps(VALIDATION,indent=2),flush=True)


if __name__=='__main__':
    main()
