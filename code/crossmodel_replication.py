from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from utility_experiment import ROOT,datasets,split_data,TreeRules,mask

OUT=ROOT/'results/crossmodel-robustness/results'; OUT.mkdir(parents=True,exist_ok=True)

def homogeneity(X,y):
    ids=NearestNeighbors(n_neighbors=6).fit(X).kneighbors(X,return_distance=False)
    knn=np.mean([np.mean(y[[j for j in neighbours if j!=i][:5]]==y[i]) for i,neighbours in enumerate(ids)])
    groups=KMeans(n_clusters=5,random_state=42,n_init=10).fit_predict(X)
    purity=np.mean([np.bincount(y[groups==g],minlength=2).max()/(groups==g).sum() for g in np.unique(groups)])
    return .7*knn+.3*purity

def forward(X,m):
    zs=[]; h=X
    for w,b in zip(m.coefs_[:-1],m.intercepts_[:-1]):
        z=h@w+b; zs.append(z); h=np.maximum(0,z)
    return zs,np.hstack([z>0 for z in zs]),h@m.coefs_[-1]+m.intercepts_[-1]

def order_gates(x,m):
    zs,_,_=forward(x[None,:],m); sens=m.coefs_[-1].ravel(); gs=[]
    for l in range(len(zs)-1,-1,-1):
        dz=sens*(zs[l][0]>0); gs.append(np.abs(dz))
        if l: sens=dz@m.coefs_[l].T
    return np.argsort(np.concatenate(gs[::-1]))[::-1]

def measures(mr,mf,p,pt,c):
    n=int(mr.sum()); nf=int(mf.sum()); precision=np.mean(p[mr]==c) if n else np.nan
    base=np.mean(p==c)
    return {'precision':precision,'fit_precision':np.mean(pt[mf]==c) if nf else np.nan,
            'coverage':n/len(p),'ref_count':n,'evaluable':int(n>0),'constant_precision':base,
            'lift':precision-base if n else np.nan}

def run():
    rows=[]; fits=[]; t0=time.time()
    with threadpool_limits(limits=1):
        for name,X,y,names in datasets():
            for seed in range(5):
                (Xf,Xs,Xr,Xq),indices,imp=split_data(X,y,seed); fi,si,ri,qi=indices
                scale=StandardScaler().fit(Xf); Xf,Xs,Xr,Xq=[scale.transform(a).astype(np.float32) for a in [Xf,Xs,Xr,Xq]]
                H=homogeneity(Xf,y[fi])
                models={'tree':DecisionTreeClassifier(max_depth=8,random_state=seed),
                        'linear':LogisticRegression(C=1,max_iter=2000,random_state=seed)}
                if name!='internet-ads': models['relu']=MLPClassifier(hidden_layer_sizes=(16,16),max_iter=1500,tol=1e-5,random_state=seed)
                for kind,m in models.items():
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always'); st=time.time(); m.fit(Xf,y[fi])
                    p=m.predict(Xr); pt=m.predict(Xf); pq=m.predict(Xq)
                    warn=[str(w.message) for w in ws]
                    fits.append({'dataset':name,'seed':seed,'model':kind,'H':H,'accuracy':np.mean(p==y[ri]),
                                 'label_majority':np.bincount(y[ri]).max()/len(ri),'model_majority':np.bincount(p,minlength=2).max()/len(p),
                                 'd':X.shape[1],'n':len(y),'fit_seconds':time.time()-st,'warnings':json.dumps(warn),
                                 'iterations':int(np.max(getattr(m,'n_iter_',0))), 'converged':int(not any('converg' in s.lower() or 'maximum iterations' in s.lower() for s in warn))})
                    if kind=='tree': tr=TreeRules(m,Xs)
                    if kind=='relu':
                        _,sr,logr=forward(Xr,m); _,sf,_=forward(Xf,m); _,sq,logq=forward(Xq,m)
                        assert np.array_equal((logr[:,0]>0).astype(int),p)
                    for j,x in enumerate(Xq):
                        c=int(pq[j]); common={'dataset':name,'seed':seed,'model':kind,'H':H,'query_index':int(qi[j]),'class':c}
                        if kind=='tree':
                            path,_,_=tr.path(x)
                            for k in [1,2,4,8]:
                                rule=path[:k]
                                rows.append(common|{'family':'tree','k':k,'actual_length':len(rule)}|measures(mask(rule,Xr),mask(rule,Xf),p,pt,c))
                        elif kind=='linear':
                            w=m.coef_[0]; b=m.intercept_[0]
                            for fam,order in [('linear_local',np.argsort(np.abs(w*x))[::-1]),('linear_coefficient',np.argsort(np.abs(w))[::-1])]:
                                for k in [1,2,4,8]:
                                    keep=order[:k]; side=int(b+x[keep]@w[keep]>=0)
                                    mr=(b+Xr[:,keep]@w[keep]>=0)==side; mf=(b+Xf[:,keep]@w[keep]>=0)==side
                                    rows.append(common|{'family':fam,'k':k,'actual_length':len(keep)}|measures(mr,mf,p,pt,c))
                        else:
                            order=order_gates(x,m)
                            for k in [1,2,4,8,32]:
                                keep=order[:k]; mr=(sr[:,keep]==sq[j,keep]).all(axis=1); mf=(sf[:,keep]==sq[j,keep]).all(axis=1)
                                rows.append(common|{'family':'relu_gate','k':k,'actual_length':len(keep)}|measures(mr,mf,p,pt,c))
            pd.DataFrame(rows).to_csv(OUT/'queries.csv',index=False);pd.DataFrame(fits).to_csv(OUT/'fits.csv',index=False)
            print(f'{name}: {len(fits)} fits; {time.time()-t0:.1f}s',flush=True)
    (OUT/'runtime.json').write_text(json.dumps({'seconds':time.time()-t0}))

if __name__=='__main__': run()
