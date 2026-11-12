"""Checkpointed sequential PongDuel rollout then BART structure learning."""
import subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'pongduel';LOG=OUT/'sequence.log'
def log(s): OUT.mkdir(parents=True,exist_ok=True); LOG.open('a',encoding='utf-8').write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}\n")
def run(script,*args): return subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],check=False).returncode
def main():
 if not (OUT/'replay.npz').exists():
  log('collection started');
  if run('pongduel_collect.py')!=0: raise RuntimeError('collection failed')
  log('collection completed')
 for i in range(21):
  if (OUT/'targets'/f'target_{i}.json').exists(): continue
  log(f'target {i} started')
  if run('pongduel_bart_target.py',i)!=0: raise RuntimeError(f'target {i} failed')
  log(f'target {i} completed')
 log('PongDuel BART structure completed')
if __name__=='__main__':main()
