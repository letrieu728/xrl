#!/usr/bin/env python3
"""Smoke test: repo's real BARTRegressor (R/dbarts via rpy2).

Synthetic data: y = X0 + X1 + noise (repo's own test pattern).
Checks: fit, predict, ATE, ITE shape (must be 500 x N for the repo HTE
assert: ndpost=125 x nchain=4), is_causal_HTE (True for X0/X1, False for
a pure-noise column if the test is discriminating).
"""
import os
import sys
from pathlib import Path

# Repo checkout: prefer the persistent workspace copy (/tmp is wiped on
# VM replacement); fall back to /tmp if the copy is absent.
_HERE = Path(__file__).resolve().parent
_WS_REPO = _HERE.parent / "repo-causal-xrl"
_TMP_REPO = Path("/tmp/causal-xrl-muc4")
sys.path.insert(0, str(_WS_REPO if _WS_REPO.is_dir() else _TMP_REPO))
import numpy as np

# Compat shim (no repo file edited): the repo's bart_regressor.py uses
# sklearn.tree._tree.DTYPE, removed in sklearn >= 1.6 (this venv has
# 1.9.1). Old DTYPE was np.float32; restore the attribute so the repo
# code runs unchanged.
import sklearn.tree._tree as _sk_tree
if not hasattr(_sk_tree, "DTYPE"):
    _sk_tree.DTYPE = np.float32

rng = np.random.default_rng(0)
n = 300
X = rng.normal(size=(n, 3))
y = X[:, 0] + X[:, 1] + rng.normal(scale=0.5, size=n)

from cxrl.lib.models import BARTRegressor

bart = BARTRegressor()
print("BARTRegressor() constructed (R dbarts loaded)")
t0 = __import__("time").perf_counter()
bart.fit(X, y)
print(f"fit done in {__import__('time').perf_counter() - t0:.1f}s")
pred = np.asarray(bart.predict(X[:5]), dtype=float).ravel()
print("predict[:5] =", np.round(pred, 3))

ite = np.asarray(bart.ITE(X[:50], 0, additive=True), dtype=float)
print("ITE shape (expect (500, 50)):", ite.shape)
assert ite.shape[0] == 500, f"HTE assert would fail: draws={ite.shape[0]}"

ate, lci, uci = [float(v) for v in bart.ATE(X[:50], 0, additive=True)]
print(f"ATE col0 (true effect ~+1 per SD): ate={ate:.3f} CI=({lci:.3f}, {uci:.3f})")

for c in range(3):
    sig = bart.is_causal_HTE(X[:100], c, additive=True)
    print(f"is_causal_HTE col{c}: {sig}")
print("SMOKE TEST PASSED — real BART works end-to-end via rpy2")
