"""Compute PongDuel action baselines and PC/FCI full wall-clock timings."""
import json,time
from pathlib import Path
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'pongduel'
def score(model,xtr,xte,ytr,yte):
 model.fit(xtr,ytr);p=model.predict(xte);return {'accuracy':float(accuracy_score(yte,p)),'f1_macro':float(f1_score(yte,p,average='macro',zero_division=0))}
def main():
 r=np.load(OUT/'replay.npz');states=r['states'];actions=r['actions'].astype(int);current=np.concatenate((states,actions.reshape(-1,1)),axis=1);x,y=current[:-1],actions[1:]
 tr,te=train_test_split(np.arange(len(y)),test_size=.4,random_state=10,stratify=y)
 baselines={'logistic_regression':score(LogisticRegression(random_state=1,max_iter=1000),x[tr],x[te],y[tr],y[te]),'decision_tree':score(DecisionTreeClassifier(random_state=1),x[tr],x[te],y[tr],y[te])}
 # Fisher-Z requires an invertible correlation matrix. PongDuel contains
 # duplicate agent/ball columns and one-hot columns whose sum is one, so retain
 # a numerically independent state basis only for PC/FCI.
 kept=[]; rank=0
 for column in range(states.shape[1]):
  candidate=states[:,kept+[column]]
  candidate_rank=np.linalg.matrix_rank(candidate-candidate.mean(axis=0),tol=1e-10)
  if candidate_rank>rank: kept.append(column);rank=candidate_rank
 reduced_states=states[:,kept]
 pc_start=time.perf_counter();pc(reduced_states,alpha=.05,indep_test='fisherz',stable=False,show_progress=False);pc_seconds=time.perf_counter()-pc_start
 fci_start=time.perf_counter();fci(reduced_states,independence_test_method='fisherz',alpha=.05,verbose=False);fci_seconds=time.perf_counter()-fci_start
 targets=[json.loads((OUT/'targets'/f'target_{i}.json').read_text()) for i in range(21)]
 result={'environment':'PongDuel-v0','seed':99,'episodes':200,'evaluation_scope':'heuristic-policy evaluation only; no RL training','baselines':baselines,'structure_seconds':{'bart_hte_isolated_target_sum':sum(x['elapsed_seconds'] for x in targets),'pc_fisherz_wall_clock_independent_state_basis':pc_seconds,'fci_fisherz_wall_clock_independent_state_basis':fci_seconds},'pc_fci_representation':{'original_state_columns':int(states.shape[1]),'independent_state_columns':kept,'independent_column_count':len(kept)},'timing_note':'PC/FCI use a numerically independent basis of state columns because full PongDuel state data has a singular correlation matrix; timings are full wall-clock time.'}
 (OUT/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
