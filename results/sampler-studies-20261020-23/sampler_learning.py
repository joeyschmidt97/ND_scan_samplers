import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,json,time,hashlib
sys.path.insert(0,'C:/Users/joesc/git')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor,as_completed
from threadpoolctl import threadpool_limits
from mgkdb_stage2 import Predictor,load,prep,clean,metric
from ND_scan_samplers.tests.benchmarks.benchmarknd.pool import PoolOracle,initial_pool_design
from ND_scan_samplers.tests.benchmarks.benchmarknd.core import Observations
from ND_scan_samplers.src.strategies import run_arm,gpr

BASE=Path(__file__).parent;OUT=BASE/'sampler-learning-20261021'
ARMS=['random','space-filling','vurs','vwrs','gpr-var','gpr-m05-var','gpr-m25-var','gpr-u50-g50']
CHECKS=[16,32,64,98,159,256]
PANEL={'gamma':{'extra':'extra1','boost':'hist15','gp':'gp05'},'omega':{'extra':'extra4','boost':'hist31','gp':'gp05','sign':'sign_experts'}}
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def acquire(oracle,arm,seed,budget):
    start=initial_pool_design(oracle,seed)
    obs=Observations(oracle,budget,oracle.dim,seed,shared=start)
    if arm=='random':
        paid=set(oracle.indices(start));available=np.array([i for i in range(len(oracle.pool)) if i not in paid])
        order=np.random.default_rng(seed).permutation(available)
        for i in order[:obs.remaining]:obs(oracle.pool[i])
    elif arm=='gpr-m25-var':gpr(obs,seed,nu=2.5)
    else:run_arm(arm,obs,seed)
    idx=oracle.indices(obs.x)
    assert len(idx)==budget and len(set(idx))==budget
    target=Observations.initial_design(oracle.dim,seed)
    distance=np.linalg.norm(start-target,axis=1)
    return idx,{'median':float(np.median(distance)),'p95':float(np.quantile(distance,.95)),'max':float(distance.max())}
def prepare():
    assert not (OUT/'manifest.json').exists(),'Refuse to replace a frozen manifest'
    OUT.mkdir(exist_ok=True);(OUT/'trajectories').mkdir(exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
    d,paths=load();dev,features,groups=prep(d)
    good=np.isfinite(dev[['omt_e','omn_e','x0','kymin']]).all(axis=1)&(dev.kymin>0)
    dev=dev[good].copy();raw=dev[['omt_e','omn_e','x0','kymin']].to_numpy(copy=True);raw[:,3]=np.log(raw[:,3])
    keys=[tuple(x) for x in np.round(raw,12)];seen={};keep=[];dropped=[]
    for i,key in enumerate(keys):
        if key in seen:dropped.append({'row':int(dev.index[i]),'kept_row':int(dev.index[seen[key]])})
        else:seen[key]=i;keep.append(i)
    dev=dev.iloc[keep];raw=raw[keep];lo=raw.min(0);hi=raw.max(0);assert (hi>lo).all()
    coords=(raw-lo)/(hi-lo);assert len(dev)>=512
    test=clean(d[d.directory.str.contains('pscratch_sd_d_drhatch_pscans2_scanfiles',regex=False)])
    assert len(test)>20 and test.directory.nunique()>=3
    labels=pd.Categorical(dev.mode_ID.fillna('Unknown').astype(str));scales=dev[['gamma','omega']].std(ddof=0).to_numpy()
    assert np.isfinite(scales).all() and (scales>0).all()
    meta={'train_rows':len(dev),'test_rows':len(test),'test_groups':int(test.directory.nunique()),'features':features,'axis_names':['omt_e','omn_e','x0','log_kymin'],'axis_low':lo.tolist(),'axis_high':hi.tolist(),'excluded_nonfinite_coordinates':int((~good).sum()),'dropped_duplicates':dropped,'train_row_ids':dev.index.tolist(),'test_row_ids':test.index.tolist(),'test_directory':test.directory.tolist(),'test_mode':test.mode_ID.fillna('Unknown').astype(str).tolist(),'scale':scales.tolist(),'panel':PANEL,'arms':ARMS,'checkpoints':CHECKS,'seeds':list(range(20)),'script_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol_hash':hashlib.sha256((BASE/'sampler-learning-protocol.md').read_bytes()).hexdigest(),'data_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'source_hash':hashlib.sha256(Path('C:/Users/joesc/git/ND_scan_samplers/src/strategies.py').read_bytes()).hexdigest()}
    write(OUT/'manifest.json',meta)
    np.savez_compressed(OUT/'data.npz',coords=coords,x=dev[features].to_numpy(),y=dev[['gamma','omega']].to_numpy(),labels=labels.codes,xt=test[features].to_numpy(),yt=test[['gamma','omega']].to_numpy(),scales=scales)
    write(OUT/'status.json',{'status':'prepared','completed':0,'planned':160})
    print('Prepared inputs and fixed manifest; no predictions scored',flush=True)
def evaluate(x,y,xt,yt,scale,seed,arm,n):
    rows=[];preds={}
    for j,target in enumerate(['gamma','omega']):
        for family,config in PANEL[target].items():
            start=time.monotonic();model=Predictor(config,seed).fit(x,y[:,j]);p=model.predict(xt)
            rows.append(dict(arm=arm,seed=seed,n=n,target=target,predictor=family,config=config,seconds=time.monotonic()-start,p95_absolute_error=float(np.quantile(np.abs(yt[:,j]-p),.95)),**metric(yt[:,j],p,scale[j])))
            preds[f'{target}_{family}']=p
    return rows,preds
def task(arg):
    seed,arm=arg
    with threadpool_limits(limits=1):
        a=np.load(OUT/'data.npz');oracle=PoolOracle(a['coords'],a['y'][:,0],a['labels'])
        start=time.monotonic();order,snap=acquire(oracle,arm,seed,256);acq=time.monotonic()-start
        rows=[]
        for n in CHECKS:
            ids=order[:n];scores,pred=evaluate(a['x'][ids],a['y'][ids],a['xt'],a['yt'],a['scales'],seed,arm,n);rows+=scores
            np.savez_compressed(OUT/'predictions'/f'{arm}-{seed}-{n}.npz',**pred)
        write(OUT/'trajectories'/f'{arm}-{seed}.json',{'seed':seed,'arm':arm,'order':order.tolist(),'snap':snap,'acquisition_seconds':acq,'rows':rows})
    return seed,arm
def reference():
    a=np.load(OUT/'data.npz')
    with threadpool_limits(limits=1):
        rows,pred=evaluate(a['x'],a['y'],a['xt'],a['yt'],a['scales'],0,'full-pool',len(a['x']))
    write(OUT/'reference.json',rows);np.savez_compressed(OUT/'predictions'/'reference.npz',**pred)
def render(complete=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for p in sorted((OUT/'trajectories').glob('*.json')):rows+=json.loads(p.read_text())['rows']
    if not rows:return
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'checkpoint_scores.csv',index=False)
    summary=frame.groupby(['target','predictor','arm','n']).nrmse.agg(['mean','std','count']).reset_index();summary.to_csv(OUT/'learning_curve_summary.csv',index=False)
    refs=json.loads((OUT/'reference.json').read_text())
    for target in ['gamma','omega']:
        families=list(PANEL[target]);fig,axes=plt.subplots(1,len(families),figsize=(5*len(families),4),squeeze=False)
        for ax,family in zip(axes[0],families):
            for arm in ARMS:
                q=summary[(summary.target==target)&(summary.predictor==family)&(summary.arm==arm)]
                if len(q):ax.plot(q.n,q['mean'],marker='o',markersize=3,label=arm)
            ref=next(x['nrmse'] for x in refs if x['target']==target and x['predictor']==family)
            ax.axhline(ref,ls='--',color='black',label='full pool');ax.set_title(family);ax.set_xlabel('Paid simulations');ax.set_ylabel('Normalized RMSE');ax.grid(alpha=.2)
        axes[0][-1].legend(fontsize=7,bbox_to_anchor=(1.02,1));fig.suptitle(f'{target}: '+('complete' if complete else 'PROVISIONAL, unequal completion counts'));fig.tight_layout();fig.savefig(OUT/f'{target}_learning_curves.png',dpi=150);plt.close(fig)
    primary=frame[frame.n.isin([32,64,98])].groupby(['target','predictor','arm','seed']).nrmse.mean().reset_index();primary.to_csv(OUT/'primary_seed_scores.csv',index=False)
    mean=primary.groupby(['target','predictor','arm']).nrmse.mean().unstack('arm');mean.to_csv(OUT/'primary_score_table.csv')
    fig,ax=plt.subplots(figsize=(12,5));im=ax.imshow(mean.to_numpy(),aspect='auto',cmap='viridis_r');ax.set_xticks(range(len(mean.columns)),mean.columns,rotation=35,ha='right');ax.set_yticks(range(len(mean)),[f'{a}/{b}' for a,b in mean.index]);fig.colorbar(im,label='Mean normalized RMSE');ax.set_title('Sampler × predictor — '+('complete' if complete else 'PROVISIONAL'));fig.tight_layout();fig.savefig(OUT/'sampler_predictor_heatmap.png',dpi=150);plt.close(fig)
    title='Complete' if complete else 'Provisional — do not rank incomplete arms'
    html=f'<html><meta charset="utf-8"><title>MGKDB sampler learning curves</title><h1>{title}</h1><p>Conditional on one archive pair; three base model families plus omega sign experts. See protocol and manifest. Tables show actual completion counts.</p>'
    for pic in ['gamma_learning_curves.png','omega_learning_curves.png','sampler_predictor_heatmap.png']:html+=f'<img style="max-width:100%" src="{pic}">'
    html+=mean.to_html()+summary.to_html(index=False)+'</html>';(OUT/'index.html').write_text(html,encoding='utf-8')
def main():
    if '--smoke' in sys.argv:
        rng=np.random.default_rng(9);coords=rng.random((80,4));oracle=PoolOracle(coords,np.sin(coords[:,0]*4)+coords[:,1],np.zeros(80,dtype=int))
        for arm in ARMS:acquire(oracle,arm,0,16)
        print('All 8 acquisition arms passed synthetic uniqueness/budget checks');return
    if '--prepare' in sys.argv:prepare();return
    if '--render' in sys.argv:render();return
    assert (OUT/'manifest.json').exists()
    if (OUT/'reference.json').exists():raise RuntimeError('Refuse overwrite: resume must be explicit in a future runner')
    write(OUT/'status.json',{'status':'running','completed':0,'planned':160})
    reference();done=0;failures=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(task,(seed,arm)):(seed,arm) for seed in range(20) for arm in ARMS}
        for f in as_completed(futures):
            try:seed,arm=f.result();done+=1;print(f'Completed {done}/160 seed={seed} arm={arm}',flush=True)
            except Exception as exc:failures.append({'task':futures[f],'error':repr(exc)});print('FAILED',failures[-1],flush=True)
            write(OUT/'status.json',{'status':'running','completed':done,'planned':160,'failures':failures})
            if done and done%8==0:render()
    render(complete=done==160 and not failures)
    write(OUT/'status.json',{'status':'complete' if done==160 and not failures else 'incomplete','completed':done,'planned':160,'failures':failures,'paired_analysis':'pending'})
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
