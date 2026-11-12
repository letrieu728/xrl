"""Wait for PongDuel BART structure then run metrics."""
import subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'pongduel';LOG=OUT/'metrics.log'
def log(m):
 with LOG.open('a',encoding='utf-8') as f:f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n")
def main():
 while not (OUT/'targets'/'target_20.json').exists():time.sleep(15)
 log('baseline, PC and FCI started')
 r=subprocess.run([sys.executable,str(ROOT/'scripts'/'pongduel_metrics.py')],check=False)
 if r.returncode or not (OUT/'metrics.json').exists():raise RuntimeError('PongDuel metrics failed')
 log('baseline, PC and FCI completed')
if __name__=='__main__':main()
