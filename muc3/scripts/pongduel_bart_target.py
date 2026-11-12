"""Infer one PongDuel target in a fresh Python/R process."""
import argparse,json,time
from pathlib import Path
import numpy as np
from cxrl.lib.causal.infer_network import infer_causes_hte
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'pongduel'
NAMES=['agent1_row','agent1_col','ball_row1','ball_col1','dir_NW1','dir_W1','dir_SW1','dir_SE1','dir_E1','dir_NE1','agent2_row','agent2_col','ball_row2','ball_col2','dir_NW2','dir_W2','dir_SW2','dir_SE2','dir_E2','dir_NE2','encoded_action']
def main():
 p=argparse.ArgumentParser();p.add_argument('target',type=int,choices=range(21));a=p.parse_args()
 r=np.load(OUT/'replay.npz'); current=np.concatenate((r['states'],r['actions'].reshape(-1,1)),axis=1);x,y=current[:-1],current[1:,a.target]
 started=time.perf_counter();causes,_=infer_causes_hte(x,y);d=OUT/'targets';d.mkdir(parents=True,exist_ok=True)
 res={'target_index':a.target,'target':NAMES[a.target],'causes':[NAMES[i] for i in causes],'cause_indices':causes,'method':'BART heterogeneous treatment effects','elapsed_seconds':time.perf_counter()-started}
 (d/f'target_{a.target}.json').write_text(json.dumps(res,indent=2),encoding='utf-8');print(json.dumps(res))
if __name__=='__main__':main()
