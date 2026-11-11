"""Surrogate for the CausalXRL BARTRegressor (R/dbarts via rpy2).

SURROGATE ESTIMATOR (RandomForest, 200 trees), NOT BART/dbarts
-- stability check only, to be re-run with real BART when available.

Why this file exists
--------------------
The CausalXRL repo (github.com/yanmluk/causal-xrl, commit ae9096e)
implements causal discovery in ``cxrl/lib/causal/infer_network.py``
(``infer_DBN``, ``infer_causes_hte`` / ``infer_causes_ate``) on top of
``cxrl/lib/models/bart_regressor.py``: ``BARTRegressor``, which wraps the
R package ``dbarts`` through ``rpy2``. R is NOT installed on this VM, so
the real BART estimator cannot run here. This module provides a
drop-in surrogate with the SAME interface that ``infer_network.py``
uses, so the repo pipeline itself runs unchanged except for the
estimator:

* ``fit(X, y)`` / ``predict(X)``
* ``ATE(X, col, additive)`` -> ``(ate, lci, uci)``
* ``ITE(X, col, additive)`` -> per-draw individual treatment effects
* ``is_causal_HTE(X, col, additive)`` -> bool (heterogeneous-effect test)
* ``CITestFast(X_train, y_train, X_test, y_test, col)`` -> ``(bool, secs)``
* ``CITest(X, y, col)``, ``is_state_causal(X, col)``,
  ``local_treatments(X)`` (kept for interface completeness)

Base learner: ``sklearn.ensemble.RandomForestRegressor``
(n_estimators=200, random_state=0, n_jobs=2). The 200 trees play the
role of BART's 500 posterior draws: ``ITE`` returns a (200, N) array of
per-tree treatment effects, and ``ATE`` / ``is_causal_HTE`` apply the
repo's draw-based math verbatim:

* ATE (repo ``BARTRegressor.ATE`` / R ``ate``): per-draw ATE =
  row-mean of ITE draws; ATE = mean over draws; 95% CI = quantiles
  [0.025, 0.975] over draws.
* HTE (repo ``HTE`` class): cluster points by mean ITE with a
  DecisionTreeRegressor (max_depth=4, min_samples_leaf=10% of N);
  a leaf cluster is significant when the pooled quantiles [0.025,
  0.975] of its ITE draws (draws x points) exclude 0; a column is
  causal when ANY cluster is significant.

CI-level sensitivity (90% vs 95%) recomputes quantiles from the SAME
stored draw arrays -- no re-run.

Why RandomForest and not HistGradientBoosting (first attempt)
------------------------------------------------------------
The first version used HistGradientBoostingRegressor with bootstrap
CIs over the estimation points (mean ITE +/- bootstrap CI). That
smoke run (50 episodes, hte) returned the COMPLETE graph, 25/25
edges: a CI of the mean only captures sampling variation of the
mean, not model uncertainty, so any systematic learned effect --
however small -- looks "significant". The repo's BART math instead
quantifies uncertainty through posterior draws, which a point
bootstrap cannot mimic. Switching to RandomForest per-tree draws
reproduces the repo's draw-based ATE/HTE math exactly, and the same
50-episode data then gives a discriminating result (hte: 6 edges,
ate: 14 edges, vs 16 direct-cause edges in the true CartPole-v1
physics). The task spec allows RandomForest as the base learner.

Documented deviations / limitations vs the repo's BARTRegressor
----------------------------------------------------------------
1. Draws are RandomForest trees (200), not BART posterior draws
   (500). Trees are correlated (bootstrap + feature randomness), so
   surrogate intervals are expected to be narrower than a BART
   posterior in some cases and wider in others; do NOT compare
   magnitudes or edge counts against BART/paper numbers.
2. Counterfactual contrast for numeric (``additive=True``) columns:
   the repo's ``ATE``/``ITE`` use ``X + 1`` vs ``X + 0`` (a +1 SD
   contrast, because ``infer_causes_*`` standardise numeric columns
   first). This surrogate uses ``X + 1`` vs ``X - 1`` (+1 SD vs -1 SD,
   a 2 SD contrast), as specified for this ablation. ATE magnitudes
   are therefore roughly 2x the repo's contrast for numeric columns;
   do NOT compare magnitudes across estimators.
3. Like all tree ensembles, RandomForest cannot extrapolate beyond
   the training range, so counterfactual shifts of +-1 SD are
   attenuated where the shifted value leaves the observed range.
4. ``train_test_split`` inside ``infer_causes_hte`` /
   ``infer_causes_ate`` uses ``random_state=None`` in the repo. The
   run script seeds the global numpy RNG (seed 0) before each run so
   splits are reproducible; the 70/30 (hte) and 50/50 (ate)
   proportions are unchanged.
5. The repo's ``HTE`` class asserts exactly 500 posterior draws; its
   logic is re-implemented here without that assertion (200 draws).

Every number produced with this estimator is a SURROGATE number and
must never be presented as a BART / CausalXRL-paper number.
"""

import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

SURROGATE_LABEL = (
    "SURROGATE ESTIMATOR (RandomForest, 200 trees), NOT BART/dbarts"
    " - stability check only, to be re-run with real BART when available"
)

N_DRAWS_DEFAULT = 200


def _as_2d_float(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    return X


def _as_1d_float(y):
    return np.asarray(y, dtype=float).ravel()


class SurrogateRegressor:
    """RandomForest stand-in for BARTRegressor.

    Same call interface as the repo's BARTRegressor where
    ``infer_network.py`` uses it. Per-tree predictions are the
    surrogate's posterior draws.
    """

    def __init__(self, n_draws=None, random_state=0):
        # n_draws=None -> module-level N_DRAWS_DEFAULT, so the run
        # script can set it before any instance is created.
        self.n_draws = (int(n_draws) if n_draws is not None
                        else int(N_DRAWS_DEFAULT))
        self.random_state = int(random_state)
        self._model = None
        self.mse = None  # cached per-sample squared errors (CITestFast)
        self.hte_per_treat = {}
        # stored draws for CI-level sensitivity (no re-run needed):
        #   _ate_draws[col] -> (n_draws,) per-draw ATEs
        #   _ate_point[col] -> (ate, lci, uci) at the fitted level
        #   _hte_draws[col] -> (n_draws, N) per-tree ITE draws
        #   _hte_clusters[col] -> (N,) leaf id per estimation point
        #   _hte_cluster_size[col] -> {leaf_id: n_points}
        self._ate_draws = {}
        self._ate_point = {}
        self._hte_draws = {}
        self._hte_clusters = {}
        self._hte_cluster_size = {}

    # ------------------------------------------------------------------
    # core fit / predict
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = _as_2d_float(X)
        y = _as_1d_float(y)
        self._model = RandomForestRegressor(
            n_estimators=self.n_draws, random_state=self.random_state,
            n_jobs=2)
        self._model.fit(X, y)
        self._n_features = X.shape[1]
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("No model has been fit.")
        return self._model.predict(_as_2d_float(X))

    def _predict_per_tree(self, X):
        """Per-tree predictions, shape (n_draws, N)."""
        X = _as_2d_float(X)
        return np.array([t.predict(X) for t in self._model.estimators_])

    # ------------------------------------------------------------------
    # counterfactual helpers
    # ------------------------------------------------------------------
    def _counterfactuals(self, X, trt_col, trt=1.0, ctrl=0.0, additive=False):
        """Build treated / control copies of X.

        Binary / action columns (additive=False): set col to 1 vs 0.
        Numeric columns (additive=True): shift col by +1 vs -1, i.e.
        +1 SD vs -1 SD in the standardised space that infer_causes_*
        produce (deviation #2 from the repo's +1 vs +0; see module doc).
        """
        X = _as_2d_float(X)
        if not isinstance(trt_col, int):
            raise TypeError("trt_col must be an integer column index.")
        if trt_col >= X.shape[1]:
            raise ValueError("trt_col is not a valid index.")
        X1 = X.copy()
        X0 = X.copy()
        if additive:
            X1[:, trt_col] += 1.0
            X0[:, trt_col] -= 1.0
        else:
            X1[:, trt_col] = 1.0
            X0[:, trt_col] = 0.0
        return X1, X0

    # ------------------------------------------------------------------
    # ITE / ATE (repo draw-based math, verbatim)
    # ------------------------------------------------------------------
    def ITE(self, X, trt_col, trt=1.0, ctrl=0.0, additive=False):
        """Per-draw individual treatment effects, shape (n_draws, N).

        Mirrors BART's ITE, which returns posterior draws (500, N);
        here each row is one tree's ITE vector.
        """
        X1, X0 = self._counterfactuals(X, trt_col, trt, ctrl, additive)
        return self._predict_per_tree(X1) - self._predict_per_tree(X0)

    def ATE(self, X, trt_col, trt=1.0, ctrl=0.0, additive=False, ci_level=0.95):
        """Average treatment effect with draw-based CI.

        Same math as the repo's R ``ate``: per-draw ATE = row-mean of
        ITE draws; ATE = mean over draws; CI = quantiles over draws.
        Returns (ate, lci, uci). Per-draw ATEs are stored in
        ``self._ate_draws[trt_col]`` for CI-level sensitivity.
        """
        ite_draws = self.ITE(X, trt_col, trt, ctrl, additive)
        pmean = ite_draws.mean(axis=1)
        alpha = 1.0 - ci_level
        ate = float(pmean.mean())
        lci, uci = np.quantile(pmean, [alpha / 2.0, 1.0 - alpha / 2.0])
        self._ate_draws[int(trt_col)] = pmean
        self._ate_point[int(trt_col)] = (ate, float(lci), float(uci))
        return ate, float(lci), float(uci)

    def ate_from_draws(self, trt_col, level=0.95):
        """Recompute (ate, lci, uci, significant) at a different CI level
        from the stored per-draw ATEs -- no re-run needed."""
        draws = self._ate_draws[int(trt_col)]
        alpha = 1.0 - level
        lci, uci = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
        ate = float(draws.mean())
        return ate, float(lci), float(uci), bool(lci > 0 or uci < 0)

    # ------------------------------------------------------------------
    # HTE (heterogeneous effects; repo HTE-class logic, verbatim
    # except the hard-coded 500-draw assertion)
    # ------------------------------------------------------------------
    def is_causal_HTE(self, X, trt_col, trt=1.0, ctrl=0.0, additive=False,
                      ci_level=0.95):
        """True if any ITE cluster has a pooled CI excluding 0.

        Clusters come from a DecisionTreeRegressor on mean ITE
        (max_depth=4, min_samples_leaf=10% of N); per cluster, the
        pooled quantiles of the ITE draws (draws x points in cluster)
        must exclude 0 -- exactly the repo's ``HTE.is_significant``.
        Draws and cluster ids are stored for CI-level sensitivity.
        """
        X = _as_2d_float(X)
        ite_draws = self.ITE(X, trt_col, trt, ctrl, additive)
        ite_mean = ite_draws.mean(axis=0)
        n = X.shape[0]
        min_samples = max(10, int(n * 0.1))
        tree = DecisionTreeRegressor(min_samples_leaf=min_samples,
                                     max_depth=4,
                                     random_state=self.random_state)
        tree.fit(X, ite_mean)
        clusters = tree.tree_.apply(np.asfortranarray(X.astype(np.float32)))
        alpha = 1.0 - ci_level
        is_sig = {}
        sizes = {}
        for cid in np.unique(clusters):
            mask = clusters == cid
            sizes[int(cid)] = int(mask.sum())
            pooled = ite_draws[:, mask]
            lci, uci = np.quantile(pooled, [alpha / 2.0, 1.0 - alpha / 2.0])
            is_sig[int(cid)] = bool(lci > 0 or uci < 0)
        self._hte_draws[int(trt_col)] = ite_draws
        self._hte_clusters[int(trt_col)] = clusters
        self._hte_cluster_size[int(trt_col)] = sizes
        self.hte_per_treat[int(trt_col)] = {
            "tree": tree, "clusters": clusters,
            "is_cls_causal": is_sig,
        }
        return any(is_sig.values())

    def hte_from_draws(self, trt_col, level=0.95):
        """Recompute per-cluster significance at another CI level from
        stored draws + cluster ids. Returns (is_causal, {cid: sig})."""
        ite_draws = self._hte_draws[int(trt_col)]
        clusters = self._hte_clusters[int(trt_col)]
        alpha = 1.0 - level
        is_sig = {}
        for cid in np.unique(clusters):
            pooled = ite_draws[:, clusters == cid]
            lci, uci = np.quantile(pooled, [alpha / 2.0, 1.0 - alpha / 2.0])
            is_sig[int(cid)] = bool(lci > 0 or uci < 0)
        return any(is_sig.values()), is_sig

    def is_state_causal(self, X, trt_col):
        """Per-row causal flag from the stored HTE clustering."""
        info = self.hte_per_treat[int(trt_col)]
        tree = info["tree"]
        clusters = tree.tree_.apply(
            np.asfortranarray(_as_2d_float(X).astype(np.float32)))
        return [bool(info["is_cls_causal"][int(c)]) for c in clusters]

    def local_treatments(self, X):
        """Per-row, per-column causal flags (repo BARTRegressor API)."""
        return np.array([self.is_state_causal(X, i)
                         for i in range(_as_2d_float(X).shape[1])]).T

    # ------------------------------------------------------------------
    # conditional-independence tests (repo 'expectation' method)
    # ------------------------------------------------------------------
    def CITestFast(self, X_train, y_train, X_test, y_test, trt_col):
        """Drop-one-column MSE comparison, mirroring the repo's logic.

        Full-model per-sample squared errors are cached in ``self.mse``
        (repo caches posterior-mean MSEs); the test returns True when
        the 95th percentile of the full-model errors is below the mean
        drop-column MSE, i.e. dropping the column hurts prediction.
        Returns (is_cause, elapsed_seconds).
        """
        t0 = time.time()
        X_train = _as_2d_float(X_train)
        y_train = _as_1d_float(y_train)
        X_test = _as_2d_float(X_test)
        y_test = _as_1d_float(y_test)
        if self.mse is None:
            full = RandomForestRegressor(
                n_estimators=self.n_draws, random_state=self.random_state,
                n_jobs=2)
            full.fit(X_train, y_train)
            self.mse = (y_test - full.predict(X_test)) ** 2
        X_train_cf = np.delete(X_train, trt_col, axis=1)
        X_test_cf = np.delete(X_test, trt_col, axis=1)
        reduced = RandomForestRegressor(
            n_estimators=self.n_draws, random_state=self.random_state,
            n_jobs=2)
        reduced.fit(X_train_cf, y_train)
        mse_cf = (y_test - reduced.predict(X_test_cf)) ** 2
        uci_f = float(np.quantile(self.mse, 0.95))
        elapsed = time.time() - t0
        return bool(uci_f < mse_cf.mean()), elapsed

    def CITest(self, X, y, trt_col):
        """50/50 split CI test: bootstrap CI on the MSE difference
        (full vs drop-column model)."""
        X = _as_2d_float(X)
        y = _as_1d_float(y)
        X_train, X_est, y_train, y_est = train_test_split(
            X, y, test_size=0.5, random_state=self.random_state)
        full = RandomForestRegressor(
            n_estimators=self.n_draws, random_state=self.random_state,
            n_jobs=2)
        full.fit(X_train, y_train)
        sse = (y_est - full.predict(X_est)) ** 2
        X_train_cf = np.delete(X_train, trt_col, axis=1)
        X_est_cf = np.delete(X_est, trt_col, axis=1)
        reduced = RandomForestRegressor(
            n_estimators=self.n_draws, random_state=self.random_state,
            n_jobs=2)
        reduced.fit(X_train_cf, y_train)
        sse_cf = (y_est - reduced.predict(X_est_cf)) ** 2
        diff = sse_cf - sse  # >0 means dropping the column hurts
        rng = np.random.default_rng(3000 + int(trt_col))
        idx = rng.integers(0, len(diff), size=(self.n_draws, len(diff)))
        draws = diff[idx].mean(axis=1)
        lci, uci = np.quantile(draws, [0.025, 0.975])
        return bool(lci > 0 or uci < 0)


# Backwards-compatible alias: the run script monkeypatches
# infer_network.BARTRegressor with this name.
BARTRegressor = SurrogateRegressor
