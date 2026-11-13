"""
retrain_voice_model.py
-----------------------
Retrain the voice model from the patched audio_features.csv (all 4 members).
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# reuse the pure-numpy primitives from train_all_models.py
exec(open(ROOT / "train_all_models.py").read().split("# ═══")[0])  # just the class defs

# --- actually, inline what we need ---
class LabelEncoder2:
    def fit_transform(self, y):
        self.classes_ = np.array(sorted(set(y)))
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self._map[v] for v in y])

class RF2:
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.rng = np.random.default_rng(random_state)
        self.trees_ = []
        self.classes_ = None
    def _gini(self, y):
        if len(y)==0: return 0
        _,c=np.unique(y,return_counts=True); p=c/len(y); return 1-np.sum(p**2)
    def _build(self, X, y, depth=0):
        classes,counts=np.unique(y,return_counts=True); maj=classes[np.argmax(counts)]
        if len(classes)==1 or depth>=8 or len(y)<4: return maj
        nf=max(1,int(np.sqrt(X.shape[1]))); fi=self.rng.choice(X.shape[1],nf,replace=False)
        bg,bf,bt=-1,None,None; pg=self._gini(y)
        for f in fi:
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            thrs=(vals[:-1]+vals[1:])/2
            if len(thrs)>10: thrs=thrs[self.rng.choice(len(thrs),10,replace=False)]
            for t in thrs:
                l=y[X[:,f]<=t]; r=y[X[:,f]>t]
                if not len(l) or not len(r): continue
                g=pg-(len(l)/len(y)*self._gini(l)+len(r)/len(y)*self._gini(r))
                if g>bg: bg,bf,bt=g,f,t
        if bf is None: return maj
        m=X[:,bf]<=bt
        return {"f":bf,"t":bt,"l":self._build(X[m],y[m],depth+1),"r":self._build(X[~m],y[~m],depth+1)}
    def _pred1(self,tree,x):
        while isinstance(tree,dict): tree=tree["l"] if x[tree["f"]]<=tree["t"] else tree["r"]
        return tree
    def fit(self,X,y):
        self.classes_=np.unique(y); n=len(X); self.trees_=[]
        for _ in range(self.n_estimators):
            idx=self.rng.choice(n,n,replace=True); self.trees_.append(self._build(X[idx],y[idx]))
        return self
    def predict(self,X):
        v=np.array([[self._pred1(t,x) for t in self.trees_] for x in X])
        return np.array([np.bincount(r.astype(int),minlength=len(self.classes_)).argmax() for r in v])
    def predict_proba(self,X):
        v=np.array([[self._pred1(t,x) for t in self.trees_] for x in X])
        return np.array([np.bincount(r.astype(int),minlength=len(self.classes_))/len(r) for r in v])

def acc(a,b): return float(np.mean(a==b))
def f1w(yt,yp):
    cs=np.unique(yt); fs,ws=[],[]
    for c in cs:
        tp=np.sum((yp==c)&(yt==c)); fp=np.sum((yp==c)&(yt!=c)); fn=np.sum((yp!=c)&(yt==c))
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f=2*p*r/(p+r) if p+r>0 else 0; fs.append(f); ws.append(np.sum(yt==c))
    return float(np.average(fs,weights=ws))
def ll(yt,yp,eps=1e-15):
    return float(-np.mean([np.log(np.clip(yp[i,c],eps,1)) for i,c in enumerate(yt)]))
def tts(X,y,ts=0.25,rs=42):
    rng=np.random.default_rng(rs); idx=np.arange(len(X)); 
    te,tr=[],[]
    for c in np.unique(y):
        ci=np.where(y==c)[0]; rng.shuffle(ci); n=max(1,int(len(ci)*ts))
        te.extend(ci[:n]); tr.extend(ci[n:])
    return X[tr],X[te],y[tr],y[te]

print("="*60)
print("Retraining Voice Model (all 4 members)")
print("="*60)

df=pd.read_csv(ROOT/"audio_features.csv")
print(f"Loaded audio_features.csv  {df.shape}")
print("Members:", df["member"].value_counts().to_dict())

meta=["member","sample_id","phrase","source_file","augmentation","is_augmented"]
fc=[c for c in df.columns if c not in meta]

approved_X=df[fc].values.astype(np.float64)
rng=np.random.default_rng(42)
denied_X=approved_X.copy()+rng.normal(0,approved_X.std(axis=0)*2.5,approved_X.shape)
rng.shuffle(denied_X)

X=np.vstack([approved_X,denied_X])
yr=np.array(["approved"]*len(approved_X)+["denied"]*len(denied_X))
le=LabelEncoder2(); y=le.fit_transform(yr)
ai=int(np.where(le.classes_=="approved")[0][0])
print(f"Classes: {list(le.classes_)}  approved_idx={ai}")

Xtr,Xte,ytr,yte=tts(X,y)
print("Training RF (100 trees)...")
clf=RF2(n_estimators=100,random_state=42)
clf.fit(Xtr,ytr)

yp=clf.predict(Xte); ypr=clf.predict_proba(Xte)
va=acc(yte,yp); vf=f1w(yte,yp); vl=ll(yte,ypr)
print(f"\nVoice Model – Test Metrics")
print(f"  Accuracy : {va:.4f}")
print(f"  F1-Score : {vf:.4f}")
print(f"  Log Loss : {vl:.4f}")

clf.fit(X,y)
bundle={"model":clf,"label_encoder":le,
        "voice_labels":{int(i):str(c) for i,c in enumerate(le.classes_)},
        "approved_class_idx":ai,"feature_cols":fc,"model_name":"Random Forest (pure-numpy)",
        "test_metrics":{"accuracy":round(va,4),"f1_weighted":round(vf,4),"log_loss":round(vl,4)}}
joblib.dump(bundle,ROOT/"models"/"voice_model.pkl")
print(f"Saved models/voice_model.pkl  (all 4 members approved)")

