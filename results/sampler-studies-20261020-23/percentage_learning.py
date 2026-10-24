import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import json,time,hashlib,shutil,sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from threadpoolctl import threadpool_limits
import sampler_learning as old
from mgkdb_stage2 import Predictor,metric

BASE=Path(__file__).parent
OUT=BASE/'percentage-learning-20261022'
SOURCE=BASE/'sampler-learning-20261021'
FRACTIONS=[.05,.1,.2,.4,.6,.8,1.]
PANEL={'gamma':['extra1','extra4','hist15','hist31','gp05'],'omega':['extra1','extra4','hist31','gp05','sign_experts']}
def family(c):return 'trees' if c.startswith('extra') or c=='sign_experts' else 'boosting' if c.startswith('hist') else 'GP'
def save(p,x):old.write(p,x)

def task(arg):
    seed,arm=arg
    a=np.load(OUT/'data.npz');total=len(a['x']);checks=[int(np.ceil(f*total)) for f in FRACTIONS]
    stem=f'{arm}-{seed}';rows=[];start=time.monotonic();scored=set()
    def score(order,n):
        ids=np.sort(order[:n]);assert len(set(ids))==n
        predictions={};t=time.monotonic()
        for j,target in enumerate(['gamma','omega']):
            for config in PANEL[target]:
                m=Predictor(config,seed).fit(a['x'][ids],a['y'][ids,j]);p=m.predict(a['xt'])
                predictions[f'{target}_{config}']=p
                rows.append(dict(seed=seed,arm=arm,n=n,percent=100*n/total,target=target,config=config,family=family(config),**metric(a['yt'][:,j],p,a['scales'][j])))
        np.savez_compressed(OUT/'predictions'/f'{stem}-{n}.npz',**predictions)
        save(OUT/'checkpoints'/f'{stem}-{n}.json',{'order':order[:n].tolist(),'rows':rows[-10:],'fit_seconds':time.monotonic()-t})
        scored.add(n)
        print(f'checkpoint {stem} n={n}/{total} elapsed={time.monotonic()-start:.1f}s',flush=True)
    oracle=old.PoolOracle(a['coords'],a['y'][:,0],a['labels'])
    class Tracking(old.Observations):
        def __call__(self,x):
            if time.monotonic()-start>8*3600:raise TimeoutError('eight-hour trajectory limit; checkpoints preserved')
            result=super().__call__(x)
            order=oracle.indices(self.x)
            for n in checks[:-1]:
                if len(order)>=n and n not in scored:score(order,n)
            return result
    initial=old.initial_pool_design(oracle,seed)
    obs=Tracking(oracle,checks[-2],oracle.dim,seed,shared=initial)
    with threadpool_limits(limits=1):
        if arm=='random':
            paid=set(oracle.indices(initial));available=np.array([i for i in range(total) if i not in paid])
            for i in np.random.default_rng(seed).permutation(available)[:obs.remaining]:obs(oracle.pool[i])
        elif arm=='gpr-m25-var':old.gpr(obs,seed,nu=2.5)
        else:old.run_arm(arm,obs,seed)
        order=oracle.indices(obs.x);assert len(order)==checks[-2] and len(set(order))==len(order)
        order=np.concatenate([order,np.setdiff1d(np.arange(total),order)])
        score(order,total)
    distances=np.linalg.norm(initial-old.Observations.initial_design(oracle.dim,seed),axis=1)
    save(OUT/'trajectories'/f'{stem}.json',{'seed':seed,'arm':arm,'order':order.tolist(),'seconds':time.monotonic()-start,'snap_median':float(np.median(distances)),'snap_p95':float(np.quantile(distances,.95)),'snap_max':float(max(distances))})
    return stem

def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for p in (OUT/'checkpoints').glob('*.json'):rows+=json.loads(p.read_text())['rows']
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(OUT/'checkpoint_scores.csv',index=False)
    summary=df.groupby(['target','arm','config','percent']).nrmse.agg(['mean','std','count']).reset_index()
    summary.to_csv(OUT/'curve_summary.csv',index=False)
    colors={'trees':'#0072B2','boosting':'#D55E00','GP':'#009E73'}
    styles={'extra1':'-','extra4':'--','sign_experts':':','hist15':'-','hist31':'--','gp05':'-'}
    done=len(list((OUT/'trajectories').glob('*.json')))
    for target in PANEL:
        fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True)
        for ax,arm in zip(axes.flat,old.ARMS):
            for config in PANEL[target]:
                q=summary[(summary.target==target)&(summary.arm==arm)&(summary.config==config)]
                if len(q):ax.plot(q.percent,q['mean'],color=colors[family(config)],ls=styles[config],marker='.',label=config)
            ax.set_title(arm);ax.set_xlabel('Acquisition pool sampled (%)');ax.set_ylabel('Held-out normalized RMSE');ax.grid(alpha=.2)
        handles=[]
        from matplotlib.lines import Line2D
        for c in PANEL[target]:handles.append(Line2D([0],[0],color=colors[family(c)],ls=styles[c],label=c))
        fig.legend(handles=handles,loc='lower center',ncol=5)
        fig.suptitle(f'{target}: {done}/160 complete trajectories; '+('complete' if done==160 else 'PROVISIONAL, unequal counts')+'\nSame 247 held-out graders at every percentage; GP predictor capped at 512 fitting rows')
        fig.tight_layout(rect=(0,.05,1,.92));fig.savefig(OUT/f'{target}_sampler_panels.png',dpi=130);plt.close(fig)
    agg=df.groupby(['target','arm','seed','percent','family']).nrmse.mean().groupby(['target','arm','seed','percent']).mean().reset_index()
    agg.to_csv(OUT/'family_balanced_seed_scores.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    rng=np.random.default_rng(20261022);br=rng.integers(0,20,(10000,20));intervals=[]
    for ax,target in zip(axes,['gamma','omega']):
        for arm in old.ARMS:
            q=agg[(agg.target==target)&(agg.arm==arm)]
            stats=[]
            for pct,sub in q.groupby('percent'):
                v=sub.sort_values('seed').nrmse.to_numpy();lo=hi=np.nan
                if len(v)==20:lo,hi=np.quantile(v[br].mean(1),[.025,.975])
                stats.append((pct,v.mean(),lo,hi));intervals.append(dict(target=target,arm=arm,percent=pct,mean=v.mean(),low=lo,high=hi,count=len(v)))
            if stats:
                z=np.array(stats);line=ax.plot(z[:,0],z[:,1],label=arm)[0];ax.fill_between(z[:,0],z[:,2],z[:,3],alpha=.1,color=line.get_color())
        ax.set_title(target);ax.set_xlabel('Acquisition pool sampled (%)');ax.set_ylabel('Family-balanced normalized RMSE');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.suptitle('Descriptive aggregate; pointwise seed-bootstrap intervals at 20 pairs');fig.tight_layout();fig.savefig(OUT/'aggregate_curves.png',dpi=130);plt.close(fig)
    pd.DataFrame(intervals).to_csv(OUT/'aggregate_summary.csv',index=False)
    html=f'<html><meta charset="utf-8"><h1>Percentage learning curves: {done}/160 complete</h1><p>Follow-up on a previously inspected grading archive. No retuning. Five configurations per target, three base families. GP fitting cap: 512. Full-pool endpoint uses identical canonical inputs across samplers. Incomplete curves are provisional.</p>'
    for name in ['gamma_sampler_panels','omega_sampler_panels','aggregate_curves']:html+=f'<img style="max-width:100%" src="{name}.png">'
    (OUT/'index.html').write_text(html+summary.to_html(index=False)+'</html>',encoding='utf-8')

def main():
    if '--render' in sys.argv:render();return
    OUT.mkdir(exist_ok=False)
    for folder in ['predictions','checkpoints','trajectories']:(OUT/folder).mkdir()
    shutil.copy2(SOURCE/'data.npz',OUT/'data.npz')
    m=json.loads((SOURCE/'manifest.json').read_text());m.update(panel=PANEL,fractions=FRACTIONS,followup=True)
    m['extension_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),BASE/'percentage-protocol.md',BASE/'mgkdb_stage2.py']}
    save(OUT/'manifest.json',m);save(OUT/'status.json',dict(status='pilot',completed=0,planned=160))
    t=time.monotonic();task((0,'random'));render()
    print(f'PILOT COMPLETE {time.monotonic()-t:.1f}s; launching remaining trajectories',flush=True)
    done=1;failures=[]
    save(OUT/'status.json',dict(status='running',completed=done,planned=160,failures=failures))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(task,(seed,arm)):(seed,arm) for seed in range(20) for arm in old.ARMS if (seed,arm)!=(0,'random')}
        for f in as_completed(futures):
            try:print('COMPLETE '+f.result(),flush=True);done+=1
            except Exception as e:failures.append(dict(task=futures[f],error=repr(e)));print('FAILED '+repr(failures[-1]),flush=True)
            save(OUT/'status.json',dict(status='running',completed=done,planned=160,failures=failures));render()
    save(OUT/'status.json',dict(status='complete' if done==160 else 'incomplete',completed=done,planned=160,failures=failures));render()

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
