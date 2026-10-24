from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(__file__).parent;S=B/'mgkdb-stage2-20261021';O=S/'report';O.mkdir(exist_ok=True)
inner=pd.DataFrame(json.loads((S/'inner_scores.json').read_text()))
outer=pd.DataFrame(json.loads((S/'outer_scores.json').read_text()))
table=inner.groupby(['target','config']).nrmse.agg(['mean','std','count']).reset_index()
table.to_csv(O/'development_model_scores.csv',index=False)
outer.drop(columns=['baseline']).to_csv(O/'nested_outer_scores.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,target in zip(axes,['gamma','omega']):
    t=table[table.target==target].sort_values('mean')
    ax.barh(t.config,t['mean'],color='#407b96');ax.invert_yaxis();ax.set_title(target);ax.set_xlabel('Mean inner-validation normalized RMSE');ax.grid(axis='x',alpha=.2)
fig.suptitle('Model search: development selection scores (lower is better)\nGP fits capped at 512 rows; not independent campaign tests',fontsize=12)
fig.tight_layout();fig.savefig(O/'model_selection.png',dpi=160);plt.close(fig)
diag=json.loads((S/'diagnosis.json').read_text());data=[]
for kind in ['mode','support_quartile']:
    for label,v in diag['groups'][kind].items():data.append({'group_type':kind,'group':label,'count':v['n'],'omega_rmse':v['omega']['rmse']})
pd.DataFrame(data).to_csv(O/'frequency_diagnosis.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,kind,title in zip(axes,['mode','support_quartile'],['By recorded mode','By distance to development inputs']):
    rows=[r for r in data if r['group_type']==kind]
    ax.bar([r['group'] for r in rows],[r['omega_rmse'] for r in rows],color='#b36845')
    for i,r in enumerate(rows):ax.text(i,r['omega_rmse']+.1,f"n={r['count']}",ha='center',fontsize=8)
    ax.set_title(title);ax.set_ylabel('Omega RMSE (archive units)');ax.set_ylim(0,max(r['omega_rmse'] for r in rows)*1.2)
axes[1].set_xlabel('Distance quartile: 0 closest, 3 farthest')
fig.suptitle('First model on inspected pscans3: diagnostic, not new validation',fontsize=11);fig.tight_layout();fig.savefig(O/'frequency_failure_breakdown.png',dpi=160);plt.close(fig)
html='<html><meta charset="utf-8"><title>MGKDB model study</title><h1>Model selection and frequency diagnosis</h1><p>Separate model-selection development scores from the earlier inspected transfer diagnostic. No pscans2 scores appear here.</p><img style="max-width:100%" src="model_selection.png"><img style="max-width:100%" src="frequency_failure_breakdown.png">'+table.to_html(index=False)+pd.DataFrame(data).to_html(index=False)+'</html>'
(O/'index.html').write_text(html,encoding='utf-8')
print(O/'index.html')
