"""Locked independent-pool benchmark. Run from any cwd; all outputs are local."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.datasets import (load_iris, load_wine, load_breast_cancer, load_digits,
                              make_classification, fetch_openml)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/heldout-utility/results'
IDS = {'banknote':1462, 'blood':1464, 'diabetes':37, 'ionosphere':59,
       'sonar':40, 'internet-ads':40978}
ANCHOR_DATA = {'breast_cancer','banknote','blood','diabetes','wine','digits-35','sonar'}


def datasets():
    raw = {'iris':load_iris(return_X_y=True), 'wine':load_wine(return_X_y=True),
           'breast_cancer':load_breast_cancer(return_X_y=True),
           'digits-35':load_digits(return_X_y=True),
           'synthetic_d8':make_classification(n_samples=500,n_features=8,n_informative=5,n_redundant=2,random_state=42),
           'synthetic_d20':make_classification(n_samples=800,n_features=20,n_informative=10,n_redundant=4,random_state=1),
           'synthetic_d50':make_classification(n_samples=1000,n_features=50,n_informative=25,n_redundant=10,random_state=2),
           'synthetic_lowsep':make_classification(n_samples=800,n_features=10,n_informative=10,n_redundant=0,class_sep=.4,random_state=3)}
    for name, did in IDS.items():
        raw[name] = fetch_openml(data_id=did, as_frame=False, return_X_y=True, parser='auto', n_retries=1)
    manifest = []
    for name, (X,y) in raw.items():
        X = np.asarray(X, dtype=float)
        encoded, classes = pd.factorize(np.asarray(y,dtype=str))
        chosen = np.argsort(np.bincount(encoded))[-2:]
        keep = np.isin(encoded,chosen)
        X, y = X[keep], (encoded[keep] == chosen[1]).astype(int)
        assert np.isfinite(X[~np.isnan(X)]).all()
        names = list(load_breast_cancer().feature_names) if name=='breast_cancer' else [f'x{i}' for i in range(X.shape[1])]
        if name=='banknote':
            # UCI source order, mapped to OpenML V1--V4; display names do not affect any fit.
            names=['variance','skewness','curtosis','entropy']
        manifest.append({'dataset':name,'n':len(y),'d':X.shape[1], 'classes':[str(classes[c]) for c in chosen],
                         'missing':int(np.isnan(X).sum()),'sha256':hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()})
        yield name, X, y, names
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2))


def split_data(X,y,seed):
    idx=np.arange(len(y))
    # Calibration is drawn first WITHOUT label stratification, independently of fitted model.
    rest,sel=train_test_split(idx,test_size=.20,random_state=3000+seed)
    rest,q=train_test_split(rest,test_size=.05/.80,stratify=y[rest],random_state=seed)
    fit,ref=train_test_split(rest,test_size=.25/.75,stratify=y[rest],random_state=1000+seed)
    if len(q)>20:
        q,_=train_test_split(q,train_size=20,stratify=y[q],random_state=4000+seed)
    q=np.sort(q)
    imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(X[fit])
    return [imp.transform(X[a]).astype(np.float32) for a in [fit,sel,ref,q]], [fit,sel,ref,q], imp


class TreeRules:
    def __init__(self,model,selection):
        self.model=model; self.t=model.tree_; self.d=selection.shape[1]
        self.labels=self.t.value.argmax(axis=2).ravel()
        self.boxes=[]; self.pure={}; self.paths={}
        def walk(node,path,lo,hi):
            self.paths[node]=path
            if self.t.children_left[node]<0:
                self.boxes.append((node,lo.copy(),hi.copy(),self.labels[node]))
                self.pure[node]={int(self.labels[node])}; return self.pure[node]
            f=self.t.feature[node]; v=self.t.threshold[node]
            lh=hi.copy(); lh[f]=min(lh[f],v)
            rl=lo.copy(); rl[f]=max(rl[f],v)
            ls=walk(self.t.children_left[node],path+[(f,'<=',v)],lo,lh)
            rs=walk(self.t.children_right[node],path+[(f,'>',v)],rl,hi)
            self.pure[node]=ls|rs; return self.pure[node]
        walk(0,[],np.full(self.d,-np.inf),np.full(self.d,np.inf))
        self.lo=np.stack([b[1] for b in self.boxes]); self.hi=np.stack([b[2] for b in self.boxes])
        self.leaflabels=np.array([b[3] for b in self.boxes])
        p=model.predict(selection)
        node_masks=model.decision_path(selection).toarray().astype(bool)
        n=node_masks.sum(axis=0)
        self.tables={}
        a=.05/(2*self.t.node_count)
        for c in [0,1]:
            s=(node_masks & (p==c)[:,None]).sum(axis=0)
            lower=np.zeros_like(n,dtype=float)
            good=s>0
            lower[good]=beta.ppf(a,s[good],n[good]-s[good]+1)
            self.tables[c]=(n,s,np.divide(s,n,out=np.zeros_like(n,dtype=float),where=n>0),lower)

    def path(self,x):
        node=0; nodes=[0]
        while self.t.children_left[node]>=0:
            f=self.t.feature[node]; v=self.t.threshold[node]
            node=self.t.children_left[node] if float(x[f])<=v else self.t.children_right[node]
            nodes.append(int(node))
        return self.paths[node],nodes,int(self.labels[node])

    def sufficient(self,rule,c):
        lo=np.full(self.d,-np.inf); hi=np.full(self.d,np.inf)
        for f,op,v in rule:
            if op=='<=': hi[f]=min(hi[f],v)
            else: lo[f]=max(lo[f],v)
        feasible=(np.maximum(self.lo,lo)<np.minimum(self.hi,hi)).all(axis=1)
        return not np.any(feasible & (self.leaflabels!=c))

    def exact_delete(self,path,c):
        rule=list(path)
        for i in range(len(rule)-1,-1,-1):
            candidate=rule[:i]+rule[i+1:]
            if self.sufficient(candidate,c): rule=candidate
        assert self.sufficient(rule,c)
        assert all(not self.sufficient(rule[:i]+rule[i+1:],c) for i in range(len(rule)))
        return rule

    def prefix(self,path,nodes,c,kind='empirical',target=.95,min_count=5):
        n,s,p,l=self.tables[c]
        for k,node in enumerate(nodes):
            if kind=='structural': ok=self.pure[node]=={c}
            elif kind=='certified': ok=self.pure[node]=={c} or (n[node]>=min_count and l[node]>=target)
            else: ok=n[node]>=min_count and p[node]>=target
            if ok: return path[:k], bool(self.pure[node]=={c})
        return path,True


def mask(rule,X):
    m=np.ones(len(X),dtype=bool)
    for f,op,v in rule:
        if op=='<=': m &= X[:,f].astype(float)<=v
        else: m &= X[:,f].astype(float)>v
    return m


def empirical_delete(path,c,X,p,target=.95):
    rule=list(path)
    for i in range(len(rule)-1,-1,-1):
        candidate=rule[:i]+rule[i+1:]; m=mask(candidate,X)
        if m.sum()>=5 and (p[m]==c).mean()>=target: rule=candidate
    return rule


def evaluate(rule,c,X,p):
    m=mask(rule,X); n=int(m.sum()); s=int((p[m]==c).sum())
    return {'ref_count':n,'ref_success':s,'coverage':n/len(X),
            'precision':s/n if n else np.nan,'evaluable':int(n>0),
            'below_target':int(s/n<.95) if n else np.nan}


def run_main():
    rows=[]; fits=[]; curves=[]; sensitivity=[]; cases=[]; splits={}
    t0=time.time()
    for name,X,y,names in datasets():
        for seed in range(10):
            (Xf,Xs,Xr,Xq),indices,imp=split_data(X,y,seed)
            fi,si,ri,qi=indices
            splits[f'{name}:{seed}']={key:val.tolist() for key,val in zip(['fit','select','reference','query'],indices)}
            for depth in [4,8]:
                model=DecisionTreeClassifier(max_depth=depth,random_state=seed).fit(Xf,y[fi])
                p=model.predict(Xr); ps=model.predict(Xs)
                start=time.perf_counter(); tr=TreeRules(model,Xs); construction=time.perf_counter()-start
                fits.append({'dataset':name,'seed':seed,'depth':depth,'n_fit':len(fi),'n_select':len(si),'n_ref':len(ri),
                             'accuracy':np.mean(p==y[ri]),'label_majority':np.bincount(y[ri]).max()/len(ri),
                             'model_majority':np.bincount(p,minlength=2).max()/len(p),'nodes':model.tree_.node_count,
                             'construction_ms':construction*1000})
                paths=[tr.path(x) for x in Xq]
                case_j=min(range(len(paths)),key=lambda j:(-len(paths[j][0]),int(qi[j])))
                for j,(path,nodes,c) in enumerate(paths):
                    common={'dataset':name,'seed':seed,'depth':depth,'query_index':int(qi[j]),'class':c,'full_length':len(path)}
                    rules={}
                    for method in ['full_path','pure_prefix','exact_delete','empirical_prefix','empirical_delete','certified_prefix','constant']:
                        st=time.perf_counter(); certified=False
                        if method=='full_path': rule=path; certified=True
                        elif method=='pure_prefix': rule,certified=tr.prefix(path,nodes,c,'structural')
                        elif method=='exact_delete': rule=tr.exact_delete(path,c); certified=True
                        elif method=='empirical_prefix': rule,certified=tr.prefix(path,nodes,c)
                        elif method=='empirical_delete': rule=empirical_delete(path,c,Xs,ps)
                        elif method=='certified_prefix': rule,certified=tr.prefix(path,nodes,c,'certified')
                        else: rule=[]
                        elapsed=(time.perf_counter()-st)*1000
                        result=evaluate(rule,c,Xr,p)
                        if certified and result['ref_count']: assert result['precision']==1
                        assert mask(rule,Xq[j:j+1])[0],(method,common)
                        rows.append(common|{'method':method,'length':len(rule),'shortened':int(len(rule)<len(path)),
                                             'extraction_ms':elapsed,'structural':int(certified)}|result)
                        rules[method]={'rule':[(names[f],op,float(v)) for f,op,v in rule],**result,'length':len(rule)}
                    for k in range(1,9):
                        result=evaluate(path[:min(k,len(path))],c,Xr,p)
                        curves.append(common|{'k':k,'actual_length':min(k,len(path))}|result)
                    for target in [.90,.95,.99]:
                        for count in [5,10,20]:
                            for kind in ['empirical','certified']:
                                rule,exact=tr.prefix(path,nodes,c,kind,target,count)
                                sensitivity.append(common|{'target':target,'min_count':count,'kind':kind,'length':len(rule),'structural':int(exact)}|evaluate(rule,c,Xr,p))
                    if name in ['breast_cancer','banknote'] and seed==0 and depth==8 and j==case_j:
                        cases.append(common|{'true_label':int(y[qi[j]]),'features':dict(zip(names,map(float,Xq[j]))),'rules':rules})
        pd.DataFrame(rows).to_csv(OUT/'queries.csv',index=False)
        pd.DataFrame(fits).to_csv(OUT/'fits.csv',index=False)
        pd.DataFrame(curves).to_csv(OUT/'paired_curves.csv',index=False)
        pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False)
        (OUT/'cases.json').write_text(json.dumps(cases,indent=2))
        (ROOT/'data/splits.json').write_text(json.dumps(splits))
        print(f'{name}: complete {len(rows)} method/query rows; {time.time()-t0:.1f}s',flush=True)
    versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','matplotlib']}
    (OUT/'runtime.json').write_text(json.dumps({'seconds':time.time()-t0,'python':sys.version,'versions':versions},indent=2))


def run_anchor():
    from anchor import anchor_tabular
    class InstrumentedExplainer(anchor_tabular.AnchorTabularExplainer):
        def add_names_to_exp(self, data_row, exp, mapping):
            exp['predicate_ids']=list(exp['feature'])
            return super().add_names_to_exp(data_row, exp, mapping)
    rows=[]; t0=time.time()
    for name,X,y,names in datasets():
        if name not in ANCHOR_DATA: continue
        for seed in range(3):
            (Xf,Xs,Xr,Xq),indices,imp=split_data(X,y,seed)
            fi,si,ri,qi=indices
            model=DecisionTreeClassifier(max_depth=8,random_state=seed).fit(Xf,y[fi])
            p=model.predict(Xr)
            explainer=InstrumentedExplainer(['0','1'],names,Xf)
            dr=explainer.disc.discretize(Xr)
            anchor_j=np.sort(np.random.default_rng(5000+seed).choice(len(Xq),size=min(5,len(Xq)),replace=False))
            for j in anchor_j:
                np.random.seed(10000+100*seed+j)
                common={'dataset':name,'seed':seed,'depth':8,'query_index':int(qi[j]),'class':int(model.predict(Xq[j:j+1])[0])}
                start=time.perf_counter()
                try:
                    exp=explainer.explain_instance(Xq[j],model.predict,threshold=.95,delta=.05,tau=.05,
                            batch_size=100,max_anchor_size=8,coverage_samples=1000)
                    _,mapping=explainer.get_sample_fn(Xq[j],model.predict)
                    m=np.ones(len(Xr),dtype=bool); names_out=[]
                    # Official package stores final original predicate ids under feature before add_names.
                    # Returned exp has per-predicate feature and discretized bounds in names.
                    # Reconstruct exact predicate identity via name matching against mapping.
                    bounds=exp.exp_map.get('names',[])
                    # Package exp_map provides actual discretized ordinal ranges (inspect in smoke test).
                    predicates=exp.exp_map.get('predicate_ids')
                    if predicates is None: raise RuntimeError('predicate ids missing: instrument wrapper required')
                    for idx in predicates:
                        f,op,v=mapping[idx]
                        if op=='leq': m &= dr[:,f]<=v
                        elif op=='geq': m &= dr[:,f]>v
                        else: m &= dr[:,f]==v
                    n=int(m.sum()); s=int((p[m]==common['class']).sum())
                    rows.append(common|{'status':'ok','length':len(predicates),'ref_count':n,'ref_success':s,'coverage':n/len(Xr),
                                         'precision':s/n if n else np.nan,'native_precision':float(exp.precision()),
                                         'native_coverage':float(exp.coverage()),'extraction_ms':1000*(time.perf_counter()-start),'rule':json.dumps(bounds)})
                except Exception as e:
                    rows.append(common|{'status':type(e).__name__+': '+str(e),'extraction_ms':1000*(time.perf_counter()-start)})
                pd.DataFrame(rows).to_csv(OUT/'anchors.csv',index=False)
                print(f'Anchors {name} seed{seed} query{j}: {rows[-1]["status"]}; {time.time()-t0:.1f}s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['main','anchor'],default='main')
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='main': run_main()
    else: run_anchor()
