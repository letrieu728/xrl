"""Continue the isolated LunarLander targets sequentially after target 1 finishes."""

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "results" / "lunarlander" / "targets"
TARGET_SCRIPT = ROOT / "scripts" / "lunarlander_bart_target.py"
LOG_PATH = ROOT / "results" / "lunarlander" / "sequence.log"


def log(message):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def main():
    deadline = time.monotonic() + 900
    waiting_for = TARGET_DIR / "target_1.json"
    log("waiting for target 1 checkpoint before continuing target 2 through 11")
    while not waiting_for.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("target 1 did not produce a checkpoint within 15 minutes")
        time.sleep(5)
    for target in range(2, 12):
        completed = TARGET_DIR / f"target_{target}.json"
        if completed.exists():
            log(f"target {target}: existing result retained")
            continue
        log(f"target {target}: started")
        completed_run = subprocess.run([sys.executable, str(TARGET_SCRIPT), str(target)], check=False)
        if completed_run.returncode != 0 or not completed.exists():
            raise RuntimeError(f"target {target} failed with return code {completed_run.returncode}")
        log(f"target {target}: completed")
    log("all remaining LunarLander targets completed")


if __name__ == "__main__":
    main()
