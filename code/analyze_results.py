"""Prespecified macro endpoints, paired hierarchical intervals, figures and audit report."""
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utility_experiment import ROOT,OUT

B=5000
METRICS=['length','precision','coverage','evaluable','below_target','shortened','extraction_ms','structural']
LABELS={'full_path':'Full path','pure_prefix':'Pure prefix','exact_delete':'Exact deletion',
        'empirical_prefix':'Empirical prefix','empirical_delete':'Empirical deletion',
        'certified_prefix':'Certified prefix','constant':'Constant rule','anchors':'Anchors'}
ORDER=list(LABELS)

def ci_table(data,group,metrics,seeds=None):
    # Each dataset is weighted equally; both depths and query inputs averaged within seed.
    ds=sorted(data.dataset.unique()); seeds=sorted(data.seed.unique()) if seeds is None else seeds
    groups=sorted(data[group].unique()); rng=np.random.default_rng(20261005)
    di=rng.integers(0,len(ds),(B,len(ds)))
    si=rng.integers(0,len(seeds),(B,len(ds),len(seeds)))
    result=[]; boot={}
    means=data.groupby(['dataset','seed',group])[metrics].mean()
    for g in groups:
        vals=np.array([[means.loc[(d,s,g)].to_numpy() for s in seeds] for d in ds])
        sampled=vals[di[:,:,None],si]
        # Nan precision is excluded explicitly; other endpoints retain all queries.
        boots=np.nanmean(np.nanmean(sampled,axis=2),axis=1)
        estimates=np.nanmean(np.nanmean(vals,axis=1),axis=0)
        for j,metric in enumerate(metrics):
            lo,hi=np.nanquantile(boots[:,j],[.025,.975])
            result.append({group:g,'metric':metric,'mean':estimates[j],'ci_lo':lo,'ci_hi':hi})
            boot[g,metric]=boots[:,j]
    return pd.DataFrame(result),boot

def main():
    data=pd.read_csv(OUT/'queries.csv'); fit=pd.read_csv(OUT/'fits.csv')
    assert len(fit)==280 and len(data.dataset.unique())==14
    key=['dataset','seed','depth','query_index']
    assert data.groupby(key).method.nunique().eq(7).all()
    exact=data[data.method.isin(['full_path','pure_prefix','exact_delete'])]
    assert exact.loc[exact.evaluable==1,'precision'].eq(1).all()
    summary,boot=ci_table(data,'method',METRICS)
    summary.to_csv(OUT/'summary_ci.csv',index=False)
    dsmeans=data.groupby(['dataset','method'])[METRICS].mean().reset_index()
    dsmeans.to_csv(OUT/'dataset_metrics.csv',index=False)
    balanced=data.groupby(['dataset','seed','method','class'])[['precision','coverage','evaluable','below_target']].mean().groupby(['dataset','seed','method']).mean().reset_index()
    bsum,_=ci_table(balanced,'method',['precision','coverage','evaluable','below_target'])
    bsum.to_csv(OUT/'balanced_summary.csv',index=False)
    deltas=[]
    for method in LABELS:
        if method not in data.method.unique() or method in ['full_path','constant']: continue
        for metric in ['length','coverage','precision','evaluable']:
            b=boot[method,metric]-boot['full_path',metric]
            point=summary[(summary.method==method)&(summary.metric==metric)]['mean'].iloc[0]-summary[(summary.method=='full_path')&(summary.metric==metric)]['mean'].iloc[0]
            lo,hi=np.nanquantile(b,[.025,.975]); deltas.append({'method':method,'metric':metric,'difference':point,'ci_lo':lo,'ci_hi':hi})
    pd.DataFrame(deltas).to_csv(OUT/'paired_differences.csv',index=False)
    sens=pd.read_csv(OUT/'sensitivity.csv')
    sens.groupby(['dataset','target','min_count','kind'])[['length','precision','coverage','evaluable','structural']].mean().groupby(['target','min_count','kind']).mean().to_csv(OUT/'sensitivity_summary.csv')
    # Matched anchor subset, including every successfully returned result and all unsuccessful native targets.
    anchors=pd.read_csv(OUT/'anchors.csv')
    assert len(anchors)==105, f'Anchors still incomplete: {len(anchors)}/105'
    anchors['method']='anchors'; anchors['evaluable']=(anchors.ref_count>0).astype(int)
    anchors['below_target']=(anchors.precision<.95).astype(float).where(anchors.evaluable==1)
    anchors['native_target_met']=anchors.native_precision>=.95
    assert anchors.status.eq('ok').all(), 'Preserve and account for anchor execution errors before analysis'
    matches=data.merge(anchors[key],on=key)
    matched=pd.concat([matches,anchors],ignore_index=True)
    msum,mb=ci_table(matched,'method',['length','precision','coverage','evaluable','below_target','extraction_ms'],seeds=[0,1,2])
    msum.to_csv(OUT/'anchors_matched_ci.csv',index=False)
    anchors.groupby('dataset')[['native_precision','precision','native_target_met','coverage','evaluable']].mean().to_csv(OUT/'anchors_native.csv')
    fitsummary=fit.groupby('dataset')[['accuracy','label_majority','model_majority','construction_ms']].mean()
    fitsummary.to_csv(OUT/'model_accuracy.csv')
    curve=pd.read_csv(OUT/'paired_curves.csv')
    curve.groupby(['dataset','seed','k'])[['precision','coverage','evaluable']].mean().reset_index().to_csv(OUT/'paired_curve_summary.csv',index=False)
    # Counterexamples to per-query monotonicity, using only adjacent positive-support pairs.
    wide=curve.pivot(index=key,columns='k',values='precision').to_numpy()
    valid=np.isfinite(wide[:,:-1])&np.isfinite(wide[:,1:])
    falls=(wide[:,1:]<wide[:,:-1]-1e-12)&valid
    diagnostics={'queries':int(data.groupby(key).ngroups),'method_query_rows':len(data),'tree_fits':len(fit),
                 'paired_adjacent_comparisons':int(valid.sum()),'precision_decreases':int(falls.sum()),
                 'anchors_queries':len(anchors),'anchors_native_target_rate':float(anchors.native_target_met.mean()),
                 'platform':platform.platform(),'cpu':platform.processor()}
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    plot(summary,dsmeans)
    print(summary.pivot(index='method',columns='metric',values='mean').round(4).to_string())
    print(pd.DataFrame(deltas).round(5).to_string(index=False))
    print('Matched Anchors:',msum.pivot(index='method',columns='metric',values='mean').round(4).to_string())
    print(json.dumps(diagnostics))

def plot(summary,ds):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    methods=[m for m in ORDER if m in ds.method.unique() and m!='constant']
    colors=['#666666','#56B4E9','#0072B2','#E69F00','#D55E00','#009E73']
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.2),constrained_layout=True)
    for ax,metric,title,scale in zip(axes,['length','coverage','below_target'],['Conditions per explanation','Reference coverage (%)','Rules below 95% precision (%)'],[1,100,100]):
        for i,(m,col) in enumerate(zip(methods,colors)):
            r=summary[(summary.method==m)&(summary.metric==metric)].iloc[0]
            ax.errorbar(i,r['mean']*scale,yerr=[[max(0,(r['mean']-r.ci_lo)*scale)],[max(0,(r.ci_hi-r['mean'])*scale)]],fmt='o',color=col,capsize=3)
        ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=40,ha='right')
        ax.set_title(title); ax.grid(axis='y',alpha=.2); ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.savefig(OUT/'utility_tradeoff.pdf');fig.savefig(OUT/'utility_tradeoff.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.2),constrained_layout=True)
    pivot=ds.pivot(index='dataset',columns='method',values='length')
    reduction=100*(pivot.full_path-pivot.exact_delete)/pivot.full_path
    reduction=reduction.sort_values();ax.barh(reduction.index,reduction.values,color='#0072B2')
    ax.set_xlabel('Exact condition reduction versus full path (%)');ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False);fig.savefig(OUT/'exact_reduction_by_dataset.pdf');fig.savefig(OUT/'exact_reduction_by_dataset.png',dpi=200);plt.close(fig)

def crossmodel():
    out=ROOT/'results/crossmodel-robustness/results'
    d=pd.read_csv(out/'queries.csv');f=pd.read_csv(out/'fits.csv')
    assert len(f)==205
    d['fit_minus_reference']=d.fit_precision-d.precision
    summary=d.groupby(['dataset','seed','family','k'])[['precision','coverage','evaluable','lift','fit_minus_reference']].mean().groupby(['dataset','family','k']).mean().groupby(['family','k']).mean()
    summary.to_csv(out/'summary.csv')
    balanced=d.groupby(['dataset','seed','family','k','class'])[['precision','coverage','evaluable','lift']].mean().groupby(['dataset','seed','family','k']).mean().groupby(['dataset','family','k']).mean().groupby(['family','k']).mean()
    balanced.to_csv(out/'balanced_summary.csv')
    fits=f.groupby(['dataset','model'])[['H','accuracy','label_majority','model_majority','d','n','converged']].mean().reset_index()
    fits.to_csv(out/'model_summary.csv',index=False)
    ds=d[d.k==1].groupby(['dataset','family'])[['precision','H','lift']].mean().reset_index()
    correlations=[]
    for fam,sub in ds.groupby('family'):
        model={'tree':'tree','linear_local':'linear','linear_coefficient':'linear','relu_gate':'relu'}[fam]
        sub=sub.merge(fits[fits.model==model],on='dataset',suffixes=('','_fit'))
        a=sub.H.to_numpy();b=sub.precision.to_numpy();r,p=pearsonr(a,b);rho,sp=spearmanr(a,b)
        controls=np.column_stack([np.ones(len(sub)),sub.label_majority,sub.model_majority,np.log(sub.d),np.log(sub.n),sub.accuracy])
        residual_a=a-controls@np.linalg.lstsq(controls,a,rcond=None)[0]
        residual_b=b-controls@np.linalg.lstsq(controls,b,rcond=None)[0]
        partial=np.corrcoef(residual_a,residual_b)[0,1]
        correlations.append({'family':fam,'n':len(sub),'pearson_r':r,'nominal_p':p,'spearman_rho':rho,'spearman_p':sp,'exploratory_partial_r':partial})
    pd.DataFrame(correlations).to_csv(out/'correlations_exploratory.csv',index=False)
    convergence=f.groupby('model').agg(fits=('seed','size'),converged=('converged','sum'),accuracy=('accuracy','mean'))
    convergence.to_csv(out/'convergence.csv')
    intervals=[]
    for fam,sub in d.groupby('family'):
        ci,_=ci_table(sub,'k',['precision','coverage','evaluable','lift','fit_minus_reference'])
        ci['family']=fam;intervals.append(ci)
    pd.concat(intervals).to_csv(out/'summary_ci.csv',index=False)
    good=d.merge(f[['dataset','seed','model','converged']],on=['dataset','seed','model'])
    good=good[good.converged==1]
    good.groupby(['dataset','seed','family','k'])[['precision','coverage','evaluable','lift']].mean().groupby(['dataset','family','k']).mean().groupby(['family','k']).mean().to_csv(out/'converged_summary.csv')
    cc=[]
    for fam,sub in good[good.k==1].groupby('family'):
        sub=sub.groupby('dataset')[['H','precision']].mean()
        r,p=pearsonr(sub.H,sub.precision);cc.append({'family':fam,'n':len(sub),'r':r,'nominal_p':p})
    pd.DataFrame(cc).to_csv(out/'converged_correlations.csv',index=False)
    df=pd.DataFrame(correlations).sort_values('nominal_p');adjusted=np.maximum.accumulate(np.minimum(1,df.nominal_p.to_numpy()*np.arange(len(df),0,-1)))
    df['holm_p']=adjusted;df.to_csv(out/'correlations_exploratory.csv',index=False)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),constrained_layout=True)
    for fam in ['tree','linear_local','linear_coefficient','relu_gate']:
        sub=summary.loc[fam].reset_index();sub=sub[sub.k<=8]
        axes[0].plot(sub.k,sub.precision,marker='o',label=fam.replace('_',' '))
        axes[1].plot(sub.k,sub.evaluable,marker='o',label=fam.replace('_',' '))
    axes[0].set_ylabel('Independent-reference precision');axes[1].set_ylabel('Rules with nonzero reference support')
    for ax in axes:ax.set_xlabel('Retained conditions k');ax.set_ylim(0,1.04);ax.set_xticks([1,2,4,8]);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8);fig.savefig(out/'paired_crossmodel.pdf');fig.savefig(out/'paired_crossmodel.png',dpi=220);plt.close(fig)
    print('Independent cross-model summary:\n',summary.round(4).to_string());print(convergence.to_string())
    print(pd.DataFrame(correlations).round(4).to_string(index=False))
    print('Convergence sensitivity:\n',pd.DataFrame(cc).round(4).to_string(index=False))

if __name__=='__main__':
    import sys
    if '--crossmodel' in sys.argv:crossmodel()
    else:main()
