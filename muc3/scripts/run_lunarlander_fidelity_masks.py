"""Run action-SCM local-mask checkpoints sequentially."""

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "lunarlander" / "fidelity_masks"
SCRIPT = ROOT / "scripts" / "lunarlander_action_local_mask.py"
LOG = ROOT / "results" / "lunarlander" / "fidelity_masks.log"


def log(message):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in range(8, 12):
        output = OUT_DIR / f"action_{target}.json"
        if output.exists():
            log(f"action {target}: existing result retained")
            continue
        log(f"action {target}: started")
        result = subprocess.run([sys.executable, str(SCRIPT), str(target)], check=False)
        if result.returncode != 0 or not output.exists():
            raise RuntimeError(f"action {target} failed with return code {result.returncode}")
        log(f"action {target}: completed")
    log("local action masks completed")


if __name__ == "__main__":
    main()
