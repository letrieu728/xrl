"""Continue PongDuel targets 6--20 after the already-running target 5 checkpoint."""
import subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'pongduel'; LOG=OUT/'sequence.log'
def log(message):
    with LOG.open('a',encoding='utf-8') as f: f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
def main():
    deadline=time.monotonic()+7200
    while not (OUT/'targets'/'target_5.json').exists():
        if time.monotonic()>deadline: raise TimeoutError('target 5 timed out')
        time.sleep(10)
    for target in range(6,21):
        if (OUT/'targets'/f'target_{target}.json').exists(): continue
        log(f'target {target} started')
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/'pongduel_bart_target.py'),str(target)],check=False)
        if result.returncode!=0 or not (OUT/'targets'/f'target_{target}.json').exists(): raise RuntimeError(f'target {target} failed')
        log(f'target {target} completed')
    log('PongDuel BART structure completed')
if __name__=='__main__': main()
