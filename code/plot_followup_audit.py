"""Reproducible publication figure for the fixed post-review GPU audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/reviewer-followup-gpu/results'
FIG=RESULT
FIG.mkdir(parents=True,exist_ok=True)
COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#777777']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':8,'axes.labelsize':8,'legend.fontsize':6.7,'pdf.fonttype':42,
    'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.16,'savefig.dpi':300})

def main():
    p=pd.read_csv(RESULT/'prefixes.csv')
    p=p[(p.k>=1)&(p.k<p.full_length)&(p.ref_count>0)]
    keys=['dataset','seed','depth','query_index']
    task=p.groupby(keys)[['q','precision']].mean().groupby(['dataset','seed']).mean().groupby('dataset').mean()
    fig,axes=plt.subplots(1,3,figsize=(7.15,2.65),layout='constrained',gridspec_kw={'width_ratios':[1,1.16,1]})
    ax=axes[0]
    ax.plot([0,1],[0,1],ls='--',color='#777777',lw=1,label='Mass lower bound')
    ax.scatter(task.q,task.precision,color=COLORS[0],s=21,zorder=3,label='Task mean')
    ax.set(xlabel='Reference mass ratio q',ylabel='Reference precision',xlim=(0,1.02),ylim=(0,1.02))
    ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.legend(loc='lower right')
    ax.text(.02,1.05,'(a) Budget conservatism',transform=ax.transAxes,weight='bold')

    case=next(z for z in json.loads((RESULT/'case_paths.json').read_text()) if z['dataset']=='banknote')
    entries=case['prefixes'][1:]
    ks=np.array([z['k'] for z in entries]); L=case['full_length']
    bottom=np.zeros(len(entries))
    ax=axes[1]
    for j in range(1,L):
        values=np.array([z['sibling_bad'][j-z['k']]/z['support'] if j>=z['k'] and j<L else 0 for z in entries])
        if values.max()==0: continue
        ax.bar(ks,values,bottom=bottom,width=.64,color=COLORS[j%len(COLORS)],label=f'Split {j+1}',zorder=3)
        bottom+=values
    ax.plot(ks,[1-z['q'] for z in entries],marker='o',ms=2.5,color='#444444',lw=1,label='Mass upper bound')
    ax.set(xlabel='Retained prefix length k',ylabel='Reference precision deficit',ylim=(0,1.02))
    ax.set_xticks(ks);ax.set_yticks([0,.5,1])
    ax.legend(loc='center left',bbox_to_anchor=(0,.57),fontsize=5.8,ncol=2,
              handlelength=1.1,columnspacing=.6,frameon=False)
    ax.text(.02,1.05,'(b) Banknote sibling losses',transform=ax.transAxes,weight='bold')

    summary=pd.read_csv(RESULT/'sibling_localization_ci.csv').set_index('metric')
    ax=axes[2]
    for i,(metric,color) in enumerate([('selected_error_share',COLORS[0]),('uniform_error_share',COLORS[1])]):
        r=summary.loc[metric]
        ax.bar(i,100*r['mean'],color=color,width=.55,zorder=3)
        ax.errorbar(i,100*r['mean'],yerr=[[100*(r['mean']-r.ci_lo)],[100*(r.ci_hi-r['mean'])]],
                    color='#333333',fmt='none',capsize=3,zorder=4)
    ax.set_xticks([0,1],['Calibration\nselected','Uniform\nreference'])
    ax.set(ylabel='Reference error mass localized (%)',ylim=(0,100));ax.set_yticks([0,50,100])
    ax.text(.02,1.05,'(c) Independent localization',transform=ax.transAxes,weight='bold')
    fig.savefig(FIG/'compression-budget-audit.pdf',bbox_inches='tight')
    fig.savefig(FIG/'compression-budget-audit.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print('Figure:',FIG/'compression-budget-audit.pdf')

if __name__=='__main__':main()
