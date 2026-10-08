"""Exhaustive path-subset controls on the existing 105 Anchors queries.

Controls optimize original threshold literals, not fixed-feature AXp sets.
No reference data are consulted during selection. Enumeration is feasible
only because fitted depth is <=8; no scalability claim is made.
"""
from itertools import combinations
import json
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utility_experiment import ROOT, OUT as OLD, datasets, split_data, TreeRules, mask, evaluate

OUT = ROOT/'experiments/matched-subset-audit/results'

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.read_csv(OLD/'anchors.csv')
    rows = []; t0 = time.time()
    for name, X, y, names in datasets():
        selected = anchor[anchor.dataset==name]
        if selected.empty: continue
        for seed in sorted(selected.seed.unique()):
            (Xf,Xs,Xr,Xq), indices, _ = split_data(X,y,int(seed))
            fi,si,ri,qi=indices
            model=DecisionTreeClassifier(max_depth=8,random_state=int(seed)).fit(Xf,y[fi])
            tr=TreeRules(model,Xs); ps=model.predict(Xs); pr=model.predict(Xr)
            ids=set(selected[selected.seed==seed].query_index)
            for j,x in enumerate(Xq):
                if qi[j] not in ids: continue
                path,nodes,c=tr.path(x)
                for kind in ['minimum_exact_subset','minimum_empirical_subset']:
                    st=time.perf_counter(); chosen=None; best_support=-1; tested=0
                    for k in range(len(path)+1):
                        for keep in combinations(range(len(path)),k):
                            rule=[path[i] for i in keep]; tested+=1
                            ms=mask(rule,Xs); support=int(ms.sum())
                            ok=tr.sufficient(rule,c) if kind=='minimum_exact_subset' else support>=5 and np.mean(ps[ms]==c)>=.95
                            if ok and support>best_support:
                                chosen=rule; best_support=support
                        if chosen is not None: break
                    if chosen is None: chosen=path
                    if kind=='minimum_exact_subset': assert tr.sufficient(chosen,c)
                    rows.append({'dataset':name,'seed':int(seed),'depth':8,'query_index':int(qi[j]),'class':c,
                                 'method':kind,'length':len(chosen),'full_length':len(path),'candidates_tested':tested,
                                 'extraction_ms':1000*(time.perf_counter()-st)}|evaluate(chosen,c,Xr,pr))
        pd.DataFrame(rows).to_csv(OUT/'queries.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['dataset','seed','method'])[['length','precision','coverage','evaluable','below_target','extraction_ms']].mean().groupby('method').mean()
    summary.to_csv(OUT/'summary.csv')
    (OUT/'diagnostics.json').write_text(json.dumps({'matched_queries':len(rows)//2,'seconds':time.time()-t0},indent=2))
    assert len(rows)==210
    print(summary.to_string())

if __name__=='__main__': run()
