from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).parent/'sampler-learning-20261021'
d=pd.read_csv(out/'primary_seed_scores.csv')
rows=[]
rng=np.random.default_rng(20261021)
boot=rng.integers(0,20,size=(10000,20))
for (target,predictor),group in d.groupby(['target','predictor']):
    wide=group.pivot(index='seed',columns='arm',values='nrmse').sort_index()
    assert list(wide.index)==list(range(20)) and not wide.isna().any().any()
    assert (wide.vurs>0).all()
    for arm in wide.columns.drop('vurs'):
        delta=(wide[arm]-wide.vurs).to_numpy()
        rel=delta/wide.vurs.to_numpy()
        lo,hi=np.quantile(rel[boot].mean(axis=1),[.025,.975])
        alo,ahi=np.quantile(delta[boot].mean(axis=1),[.025,.975])
        p=1. if np.all(delta==0) else float(wilcoxon(delta,alternative='two-sided',method='auto').pvalue)
        rows.append(dict(target=target,predictor=predictor,comparator=arm,pairs=20,mean_difference=delta.mean(),absolute_ci_low=alo,absolute_ci_high=ahi,mean_relative_difference=rel.mean(),relative_ci_low=lo,relative_ci_high=hi,p_value=p,primary=predictor!='sign'))
r=pd.DataFrame(rows)
idx=r.index[r.primary]; assert len(idx)==42
ordered=r.loc[idx].sort_values('p_value').index
adjusted=np.minimum(1,np.maximum.accumulate(r.loc[ordered,'p_value'].to_numpy()*np.arange(42,0,-1)))
r.loc[ordered,'holm_p']=adjusted
r['decision']='No registered practical directional claim'
r.loc[r.primary & (r.holm_p<.05) & (r.relative_ci_low>.1),'decision']='VURS lower error by practical rule'
r.loc[r.primary & (r.holm_p<.05) & (r.relative_ci_high<-.1),'decision']='Comparator lower error by practical rule'
r.loc[~r.primary,'decision']='Secondary descriptive only'
r.to_csv(out/'paired_contrasts.csv',index=False)
fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True)
for i,target in enumerate(['gamma','omega']):
    for j,predictor in enumerate(['extra','boost','gp']):
        a=axes[i,j]; s=r[(r.target==target)&(r.predictor==predictor)]
        m=s.mean_relative_difference.to_numpy()*100
        a.errorbar(m,range(len(s)),xerr=np.vstack([m-s.relative_ci_low.to_numpy()*100,s.relative_ci_high.to_numpy()*100-m]),fmt='o',capsize=3)
        a.set_yticks(range(len(s)),s.comparator);a.axvline(0,color='black',lw=.8)
        a.axvspan(-10,10,color='gray',alpha=.15);a.set_title(target+' / '+predictor)
        a.grid(axis='x',alpha=.25);a.set_xlabel('Comparator error minus VURS (%)')
fig.suptitle('20 paired seeds: mean relative error difference and 95% bootstrap CI\nPositive favors VURS; gray band = practical margin; intervals are pointwise')
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/'paired_contrasts.png',dpi=160);plt.close(fig)
text='<h1>Registered paired analysis</h1><p>42 primary contrasts; 20 paired seeds; 10,000 paired bootstrap resamples; two-sided Wilcoxon with Holm correction. Positive differences favor VURS. A directional claim requires the pointwise 95% relative interval beyond 10% and Holm p &lt; .05. Seed uncertainty is conditional on this archive pair; grading-directory resampling remains pending. No equivalence follows from a failed superiority test.</p><img style="max-width:100%" src="paired_contrasts.png"><p><a href="paired_contrasts.csv">Download contrast table</a></p>'
(out/'paired_analysis.html').write_text('<html><meta charset="utf-8">'+text+r.to_html(index=False,float_format=lambda x:f'{x:.5g}')+'</html>',encoding='utf-8')
index=out/'index.html';html=index.read_text(encoding='utf-8')
if 'paired_analysis.html' not in html:index.write_text(html+'<p><a href="paired_analysis.html">Registered paired analysis and uncertainty plot</a></p>',encoding='utf-8')
status=json.loads((out/'status.json').read_text());status['paired_analysis']='complete';status['grading_group_resampling']='pending';(out/'status.json').write_text(json.dumps(status,indent=2))
print(r[r.primary].decision.value_counts().to_string())
print(r[r.primary & (r.decision!='No registered practical directional claim')][['target','predictor','comparator','mean_relative_difference','holm_p','decision']].to_string(index=False))
