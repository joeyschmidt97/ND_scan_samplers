import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import hashlib, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'mgkdb-regression-20261020'
DATA=Path('C:/Users/joesc/git/Fusion_Microinstability_Classifier/data')
FEATURES='kymin x0 beta coll q0 shat minor_r major_R rhostar omn_i omt_i temp_i dens_i omn_e omt_e temp_e dens_e omn_c omt_c temp_c dens_c'.split()
CONFIGS=[('mean',{}),('ridge01',{'alpha':.1}),('ridge10',{'alpha':10}),('ridge100',{'alpha':100}),('extra1',{'min_samples_leaf':1}),('extra4',{'min_samples_leaf':4}),('hist15',{'max_leaf_nodes':15}),('hist31',{'max_leaf_nodes':31})]

def save(name,obj):
    p=OUT/name
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(p)

def model(name,params,seed):
    if name=='mean': m=DummyRegressor()
    elif name.startswith('ridge'): m=Ridge(**params)
    elif name.startswith('extra'): m=ExtraTreesRegressor(n_estimators=200,max_features=1.,n_jobs=1,random_state=seed,**params)
    else: m=HistGradientBoostingRegressor(learning_rate=.05,max_iter=250,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=seed,**params)
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),m)

def fit_score(name,params,seed,x,y,xt,yt):
    mu=y.mean(axis=0); scale=y.std(axis=0)
    if np.any(scale<=0): raise ValueError('Constant training target')
    predictions=[]; metrics={}
    for j,target in enumerate(['gamma','omega']):
        m=model(name,params,seed)
        m.fit(x,(y[:,j]-mu[j])/scale[j])
        p=m.predict(xt)*scale[j]+mu[j]
        predictions.append(p)
        rmse=float(np.sqrt(mean_squared_error(yt[:,j],p)))
        metrics[target]={'rmse':rmse,'mae':float(mean_absolute_error(yt[:,j],p)),'r2':float(r2_score(yt[:,j],p)),'nrmse':rmse/float(scale[j])}
    metrics['joint_nrmse']=float(np.mean([metrics[t]['nrmse'] for t in ['gamma','omega']]))
    return metrics,np.column_stack(predictions)

def main():
    OUT.mkdir(exist_ok=False)
    paths=sorted(DATA.glob('mode_ID_GENE_data_chunk_*.pkl'))
    d=pd.concat([pd.read_pickle(p) for p in paths],ignore_index=True)
    dev=d[d.directory.str.contains('global_cfs_cdirs_m2116_hatch_pscans_scanfiles',regex=False)].copy()
    test=d[d.directory.str.contains('pscratch_sd_d_drhatch_pscans3_scanfiles',regex=False)].copy()
    audit={'raw_dev':len(dev),'raw_test':len(test)}
    for frame in (dev,test):
        for c in FEATURES+['gamma','omega']: frame[c]=pd.to_numeric(frame[c],errors='coerce')
        frame[FEATURES]=frame[FEATURES].replace([np.inf,-np.inf],np.nan)
    def clean(frame): return frame[(frame.gamma!=-1)&np.isfinite(frame.gamma)&np.isfinite(frame.omega)].copy()
    dev,test=clean(dev),clean(test)
    audit.update(valid_dev=len(dev),valid_test=len(test))
    features=[c for c in FEATURES if dev[c].notna().any()]
    def signatures(frame): return pd.util.hash_pandas_object(frame[features],index=False).to_numpy()
    dev=dev[~np.isin(signatures(dev),signatures(test))].copy().reset_index(drop=True)
    parent={s:s for s in dev.directory.unique()}
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    seen={}
    for sig,directory in zip(signatures(dev),dev.directory):
        if sig in seen: parent[find(directory)]=find(seen[sig])
        else: seen[sig]=directory
    groups=np.array([find(s) for s in dev.directory])
    audit.update(final_dev=len(dev),groups=len(set(groups)),features=features)
    if len(set(groups))<5 or len(test)<2: raise ValueError(f'Insufficient split support: {audit}')
    save('manifest.json',dict(audit=audit,configs=CONFIGS,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,python=sys.version,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    x=dev[features].to_numpy(); y=dev[['gamma','omega']].to_numpy()
    rows=[]; splits=[]
    for seed in range(20):
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(x,groups=groups))
        assert not set(groups[tr])&set(groups[va])
        splits.append(dict(seed=seed,train=tr.tolist(),validation=va.tolist()))
        save('splits.json',splits)
        for name,params in CONFIGS:
            start=time.monotonic()
            metrics,_=fit_score(name,params,seed,x[tr],y[tr],x[va],y[va])
            rows.append(dict(seed=seed,config=name,seconds=time.monotonic()-start,**metrics))
            save('development_scores.json',rows)
            print(f'seed={seed} config={name} complete',flush=True)
    means={name:float(np.mean([r['joint_nrmse'] for r in rows if r['config']==name])) for name,_ in CONFIGS}
    best=min(means.values())
    selected=next(name for name,_ in CONFIGS if means[name]<=best+1e-6)
    save('selection.json',dict(selected=selected,mean_development_joint_nrmse=means,rule='minimum mean validation, not a confirmatory winner'))
    xt=test[features].to_numpy(); yt=test[['gamma','omega']].to_numpy()
    results={}
    for name in dict.fromkeys([selected,'mean']):
        metrics,pred=fit_score(name,dict(CONFIGS)[name],0,x,y,xt,yt)
        results[name]=metrics
        np.savez_compressed(OUT/f'holdout_{name}.npz',truth=yt,prediction=pred,row_index=test.index.to_numpy())
    save('holdout_scores.json',results)
    save('status.json',{'status':'complete','selected':selected})
    print('COMPLETE',flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(limits=1): main()
    except Exception as exc:
        if OUT.exists(): save('status.json',{'status':'failed','error':repr(exc)})
        raise
