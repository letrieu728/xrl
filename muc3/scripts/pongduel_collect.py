"""Collect the 200-episode evaluation replay using the notebook's heuristic PongDuel policy."""
import json
from pathlib import Path
import gym
import ma_gym
import numpy as np
from gym import spaces

OUT = Path(__file__).resolve().parents[1] / "results" / "pongduel"

class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env); self.observation_space=spaces.Box(0.,1.,shape=(20,),dtype=np.float32); self.action_space=spaces.Discrete(9)
    def reset(self): return np.concatenate([np.asarray(v,dtype=np.float32) for v in self.env.reset()])
    def step(self,a):
        obs,reward,done,info=self.env.step([int(a)//3,int(a)%3])
        return np.concatenate([np.asarray(v,dtype=np.float32) for v in obs]), reward[0], all(done), info

DIRECTIONS=[(-1,-1),(0,-1),(1,-1),(1,1),(0,1),(-1,1)]
def reflect(row):
    x=(row-1)%74
    return 1+x if x<=37 else 38-(x-37)
def strong(obs):
    ar,ac,br,bc=obs[0]*40,obs[1]*30,obs[2]*40,obs[3]*30; dr,dc=DIRECTIONS[int(np.argmax(obs[4:]))]
    left=ac<15; contact=ac+1 if left else ac-1; toward=dc<0 if left else dc>0
    target=reflect(br+dr*((contact-bc)/dc)) if toward and dc and (contact-bc)/dc>0 else 20
    return 1 if target<ar-.75 else 2 if target>ar+.75 else 0
def weak(obs):
    ar,br,bc=obs[0]*40,obs[2]*40,obs[3]*30
    if bc<16.5:return 0
    return 1 if br<ar-6 else 2 if br>ar+6 else 0
def policy(state): return strong(state[:10])*3+weak(state[10:])

def main():
    np.random.seed(99); env=Wrapper(gym.make("PongDuel-v0")); env.seed(99)
    states=[]; actions=[]; rewards=[]; next_states=[]; lengths=[]
    for _ in range(200):
        s=env.reset(); done=False; n=0
        while not done:
            a=policy(s); ns,r,done,_=env.step(a)
            if not done: states.append(s);actions.append(a);rewards.append(r);next_states.append(ns)
            s=ns;n+=1
        lengths.append(n)
    env.close(); OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'replay.npz',states=np.asarray(states),actions=np.asarray(actions),rewards=np.asarray(rewards),next_states=np.asarray(next_states))
    meta={'environment':'PongDuel-v0','seed':99,'episodes':200,'policy':'repository notebook strong/weak heuristic joint policy','transitions':len(states),'episode_length_mean':float(np.mean(lengths))}
    (OUT/'collection_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta))
if __name__=='__main__':main()
