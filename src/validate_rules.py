"""Check the exact verifier against exhaustive threshold-cell representatives."""
import itertools
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from utility_experiment import TreeRules,mask,OUT

rng=np.random.default_rng(809)
X=rng.uniform(-2,2,(600,2)).astype(np.float32)
y=((X[:,0]*X[:,1]>0) ^ (X[:,0]>.9)).astype(int)
model=DecisionTreeClassifier(max_depth=5,random_state=8).fit(X,y)
tr=TreeRules(model,X[:100])
axis=[]
for f in range(2):
    vals=sorted(set(model.tree_.threshold[model.tree_.feature==f]))
    vals=[-3]+vals+[3]
    axis.append([.5*(a+b) for a,b in zip(vals[:-1],vals[1:])]+vals[1:-1])
grid=np.array(list(itertools.product(*axis)),dtype=np.float32)
pg=model.predict(grid); checked=0
for x in X[:60]:
    path,nodes,c=tr.path(x)
    assert model.predict(x[None,:])[0]==c
    for bits in itertools.product([False,True],repeat=len(path)):
        rule=[r for r,b in zip(path,bits) if b]
        observed=np.all(pg[mask(rule,grid)]==c)
        assert observed==tr.sufficient(rule,c),(path,bits)
        checked+=1
    short=tr.exact_delete(path,c)
    assert np.all(pg[mask(short,grid)]==c)
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'validation.json').write_text(json.dumps({'exhaustive_subset_checks':checked,'threshold_grid_size':len(grid),'status':'passed'},indent=2))
print(f'Exact verifier passed {checked} subsets on {len(grid)} threshold-cell representatives.')
