import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import json,sys,time,hashlib,warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier,HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,ConstantKernel
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import joblib
from mgkdb_regression import FEATURES,DATA

BASE=Path(__file__).parent
OUT=BASE/'mgkdb-stage2-20261021'
NAMES=['extra1','extra4','extra4_sub','hist15','hist31','asinh_extra4','asinh_hist15','gp05','gp15','gp25']
def save(name,obj):
    p=OUT/name; tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def load():
    paths=sorted(DATA.glob('mode_ID_GENE_data_chunk_*.pkl'))
    d=pd.concat([pd.read_pickle(p) for p in paths],ignore_index=True)
    for c in FEATURES+['gamma','omega']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d[FEATURES]=d[FEATURES].replace([np.inf,-np.inf],np.nan)
    return d,paths
def clean(d): return d[(d.gamma!=-1)&np.isfinite(d.gamma)&np.isfinite(d.omega)].copy()
def sig(d,features):return pd.util.hash_pandas_object(d[features],index=False).to_numpy()
def prep(d):
    dev=clean(d[d.directory.str.contains('global_cfs_cdirs_m2116_hatch_pscans_scanfiles',regex=False)])
    features=[c for c in FEATURES if dev[c].notna().any()]
    hold=d[d.directory.str.contains('pscratch_sd_d_drhatch_pscans2_scanfiles|pscratch_sd_d_drhatch_pscans3_scanfiles',regex=True)]
    dev=dev[~np.isin(sig(dev,features),sig(hold,features))].copy()
    parent={v:v for v in dev.directory.unique()}
    def find(a):
        while a!=parent[a]: parent[a]=parent[parent[a]];a=parent[a]
        return a
    seen={}
    for h,g in zip(sig(dev,features),dev.directory):
        if h in seen:parent[find(g)]=find(seen[h])
        else:seen[h]=g
    groups=np.array([find(g) for g in dev.directory])
    assert len(set(groups))>=5
    return dev,features,groups
def metric(y,p,scale):
    return dict(rmse=float(np.sqrt(mean_squared_error(y,p))),mae=float(mean_absolute_error(y,p)),r2=float(r2_score(y,p)),nrmse=float(np.sqrt(mean_squared_error(y,p))/scale))
class Predictor:
    def __init__(self,name,seed):self.name=name;self.seed=seed
    def fit(self,x,y):
        self.preprocessing=make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
        x=self.preprocessing.fit_transform(x)
        self.scale=float(np.std(y));assert self.scale>0
        self.center=float(np.mean(y));self.signed=self.name.startswith('asinh_')
        z=np.arcsinh(y/self.scale) if self.signed else (y-self.center)/self.scale
        name=self.name.replace('asinh_','')
        if name=='sign_experts':
            labels=(y>0).astype(int)
            self.gate=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=4,n_jobs=1,random_state=self.seed).fit(x,labels)
            self.experts={}
            for c in self.gate.classes_:
                mask=labels==c
                self.experts[int(c)]=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=4,n_jobs=1,random_state=self.seed).fit(x[mask],z[mask])
        else:
            if name.startswith('extra'):
                m=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=1 if name=='extra1' else 4,max_features=.75 if name.endswith('sub') else 1.,n_jobs=1,random_state=self.seed)
            elif name.startswith('hist'):
                m=HistGradientBoostingRegressor(max_leaf_nodes=int(name[4:]),max_iter=200,learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=self.seed)
            else:
                nu={'gp05':.5,'gp15':1.5,'gp25':2.5}[name]
                m=GaussianProcessRegressor(kernel=ConstantKernel(1.,(.01,100))*Matern(np.sqrt(x.shape[1]),(.1,100),nu=nu),alpha=1e-4,n_restarts_optimizer=0,random_state=self.seed)
                if len(x)>512:
                    ix=np.sort(np.random.default_rng(self.seed).choice(len(x),512,replace=False));x=x[ix];z=z[ix]
            with warnings.catch_warnings(record=True):m.fit(x,z)
            self.estimator=m
        return self
    def predict(self,x):
        x=self.preprocessing.transform(x)
        if self.name=='sign_experts':
            prob=self.gate.predict_proba(x);z=sum(prob[:,j]*self.experts[int(c)].predict(x) for j,c in enumerate(self.gate.classes_))
        else:z=self.estimator.predict(x)
        p=np.sinh(z)*self.scale if self.signed else z*self.scale+self.center
        assert np.isfinite(p).all()
        return p
def diagnose(d):
    old=BASE/'mgkdb-regression-20261020'
    manifest=json.loads((old/'manifest.json').read_text());features=manifest['audit']['features']
    dev=clean(d[d.directory.str.contains('global_cfs_cdirs_m2116_hatch_pscans_scanfiles',regex=False)])
    a=np.load(old/'holdout_extra1.npz');test=d.loc[a['row_index']].copy();truth=a['truth'];pred=a['prediction']
    assert np.allclose(test[['gamma','omega']],truth)
    im=make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
    x=im.fit_transform(dev[features]);xt=im.transform(test[features])
    dist=NearestNeighbors(n_neighbors=1).fit(x).kneighbors(xt)[0][:,0]
    quart=pd.qcut(dist,4,labels=False,duplicates='drop')
    buckets={'mode':test.mode_ID.fillna('Unknown').astype(str).to_numpy(),'radius':test.x0.round(3).astype(str).to_numpy(),'omega_sign':np.where(truth[:,1]>0,'positive',np.where(truth[:,1]<0,'negative','zero')),'support_quartile':quart}
    result={'counts':{'dev':len(dev),'diagnostic_test':len(test)},'groups':{},'features':{}}
    for kind,labels in buckets.items():
        result['groups'][kind]={}
        for label in np.unique(labels):
            mask=labels==label
            if mask.sum()<2:continue
            result['groups'][kind][str(label)]={'n':int(mask.sum()),**{t:metric(truth[mask,j],pred[mask,j],float(dev[t].std(ddof=0))) for j,t in enumerate(['gamma','omega'])}}
    for j,t in enumerate(['gamma','omega']):
        err=(truth[:,j]-pred[:,j])**2;top=max(1,int(np.ceil(.1*len(err))))
        result[t+'_top10_squared_error_share']=float(np.sort(err)[-top:].sum()/err.sum())
    result['omega_sign_agreement']=float(np.mean(np.sign(truth[:,1])==np.sign(pred[:,1])))
    for c in features:
        result['features'][c]={'dev_missing':float(dev[c].isna().mean()),'test_missing':float(test[c].isna().mean()),'dev_unique':int(dev[c].nunique()),'test_outside_dev_range':float(((test[c]<dev[c].min())|(test[c]>dev[c].max())).mean())}
    tmp=dev.assign(signature=sig(dev,features))
    duplicate=tmp.groupby('signature').agg(n=('gamma','size'),gamma_range=('gamma',lambda x:x.max()-x.min()),omega_range=('omega',lambda x:x.max()-x.min()))
    result['duplicate_inputs']={'groups':int((duplicate.n>1).sum()),'disagreeing':int(((duplicate.n>1)&((duplicate.gamma_range>1e-8)|(duplicate.omega_range>1e-8))).sum())}
    save('diagnosis.json',result)
    print('Diagnosis saved',flush=True)
def run():
    OUT.mkdir(exist_ok=False)
    save('status.json',{'status':'running','external_holdout_scored':False})
    d,paths=load();diagnose(d)
    dev,features,groups=prep(d)
    save('manifest.json',{'rows':len(dev),'groups':len(set(groups)),'features':features,'row_ids':dev.index.tolist(),'group_ids':groups.tolist(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'candidates_gamma':NAMES,'candidates_omega':NAMES+['sign_experts'],'reserved':'pscratch_sd_d_drhatch_pscans2; targets not scored'})
    x=dev[features].to_numpy();y=dev[['gamma','omega']].to_numpy()
    inner=[];outer=[];splits=[]
    for seed in range(20):
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(x,groups=groups))
        assert not set(groups[tr])&set(groups[va])
        folds=list(GroupKFold(3).split(x[tr],groups=groups[tr]))
        splits.append({'seed':seed,'outer_train':tr.tolist(),'outer_validation':va.tolist(),'inner':[{'train':tr[a].tolist(),'validation':tr[b].tolist()} for a,b in folds]});save('splits.json',splits)
        for j,target in enumerate(['gamma','omega']):
            names=NAMES+(['sign_experts'] if target=='omega' else [])
            for name in names:
                for fold,(a,b) in enumerate(folds):
                    ti,vi=tr[a],tr[b];assert not set(groups[ti])&set(groups[vi])
                    start=time.monotonic();m=Predictor(name,seed).fit(x[ti],y[ti,j])
                    inner.append(dict(seed=seed,target=target,config=name,fold=fold,seconds=time.monotonic()-start,**metric(y[vi,j],m.predict(x[vi]),np.std(y[ti,j]))))
                save('inner_scores.json',inner)
                print(f'seed={seed} target={target} config={name} inner complete',flush=True)
            scores={n:np.mean([r['nrmse'] for r in inner if r['seed']==seed and r['target']==target and r['config']==n]) for n in names}
            best=min(scores,key=scores.get);m=Predictor(best,seed).fit(x[tr],y[tr,j]);pred=m.predict(x[va])
            outer.append(dict(seed=seed,target=target,selected=best,**metric(y[va,j],pred,np.std(y[tr,j])),baseline=metric(y[va,j],np.full(len(va),np.mean(y[tr,j])),np.std(y[tr,j]))))
            np.savez_compressed(OUT/f'outer_{seed}_{target}.npz',row_ids=dev.index.to_numpy()[va],truth=y[va,j],prediction=pred)
            save('outer_scores.json',outer)
    selected={}
    for j,target in enumerate(['gamma','omega']):
        names=NAMES+(['sign_experts'] if target=='omega' else [])
        scores={n:float(np.mean([r['nrmse'] for r in inner if r['target']==target and r['config']==n])) for n in names}
        selected[target]={'config':min(scores,key=scores.get),'inner_scores':scores}
    save('selected_models.json',selected)
    for j,target in enumerate(['gamma','omega']):joblib.dump(Predictor(selected[target]['config'],0).fit(x,y[:,j]),OUT/f'{target}_model.joblib')
    save('status.json',{'status':'complete','external_holdout_scored':False})
    print('COMPLETE: external holdout remains sealed',flush=True)
if __name__=='__main__':
    sys.modules['mgkdb_stage2']=sys.modules[__name__]
    Predictor.__module__='mgkdb_stage2'
    with threadpool_limits(limits=1):
        if '--smoke' in sys.argv:
            rng=np.random.default_rng(8);x=rng.normal(size=(40,4));y=x[:,0]+.2*x[:,1]**2
            for name in NAMES+['sign_experts']:
                p=Predictor(name,0).fit(x[:30],y[:30]).predict(x[30:]);assert p.shape==(10,)
            print('All 11 predictors passed synthetic shape/finite checks; no MGKDB scores inspected')
        else:
            try:run()
            except Exception as exc:
                if OUT.exists():save('status.json',{'status':'failed','error':repr(exc)})
                raise
