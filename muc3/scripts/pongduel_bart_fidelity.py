"""Documented PongDuel BART numeric-action fidelity proxy.

PongDuel has no fidelity notebook and its joint action has nine categories.
The repository exposes only BART regression, therefore this reports a rounded
numeric-action surrogate, not a categorical BART reproduction.
"""
import json,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from cxrl.lib.causal.infer_network import infer_causes_hte
from cxrl.lib.models import BARTRegressor
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'pongduel'
def main():
 r=np.load(OUT/'replay.npz');current=np.concatenate((r['states'],r['actions'].reshape(-1,1)),axis=1);x,y=current[:-1],r['actions'][1:].astype(float)
 started=time.perf_counter();causes,action_scm=infer_causes_hte(x,y)
 local=action_scm.local_treatments(x)
 # As the global graph has a single action cause, retain rows where that cause
 # is locally active; fallback is all rows if no such row exists.
 active=np.any(local[:,causes],axis=1) if causes else np.zeros(len(x),dtype=bool)
 indices=np.flatnonzero(active)
 if len(indices)<20: indices=np.arange(len(x))
 train,test=train_test_split(indices,test_size=.4,random_state=10,stratify=y[indices].astype(int))
 model=BARTRegressor();model.fit(x[train],y[train]);raw=np.asarray(model.predict(x[test]))
 pred=np.clip(np.rint(raw),0,8).astype(int);truth=y[test].astype(int)
 result={'environment':'PongDuel-v0','method':'documented numeric-action BART proxy; rounded regression output for 9 joint classes','not_exact_paper_fidelity':True,'global_action_causes':causes,'local_candidate_samples':int(len(indices)),'test_samples':int(len(test)),'accuracy':float(accuracy_score(truth,pred)),'f1_macro':float(f1_score(truth,pred,average='macro',zero_division=0)),'elapsed_seconds':time.perf_counter()-started}
 (OUT/'bart_fidelity.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
