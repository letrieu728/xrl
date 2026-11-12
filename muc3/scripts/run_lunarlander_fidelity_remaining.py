"""Continue fidelity scoring with actions 9--11 after externally started action 8."""

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lunarlander"
SCORES = OUT / "fidelity_scores"
ACTION_SCRIPT = ROOT / "scripts" / "lunarlander_fidelity_action_scores.py"
AGGREGATE_SCRIPT = ROOT / "scripts" / "lunarlander_fidelity_aggregate.py"
LOG = OUT / "fidelity_scores.log"


def log(message):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def main():
    deadline = time.monotonic() + 1800
    while not (SCORES / "action_8.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("action 8 score did not complete within 30 minutes")
        time.sleep(5)
    for action in range(9, 12):
        result_path = SCORES / f"action_{action}.json"
        if result_path.exists():
            log(f"action {action}: existing score retained")
            continue
        log(f"action {action}: started")
        result = subprocess.run([sys.executable, str(ACTION_SCRIPT), str(action)], check=False)
        if result.returncode != 0 or not result_path.exists():
            raise RuntimeError(f"action {action} score failed with return code {result.returncode}")
        log(f"action {action}: completed")
    result = subprocess.run([sys.executable, str(AGGREGATE_SCRIPT)], check=False)
    if result.returncode != 0 or not (OUT / "bart_fidelity.json").exists():
        raise RuntimeError("fidelity aggregation failed")
    log("all action scores and fidelity aggregation completed")


if __name__ == "__main__":
    main()
