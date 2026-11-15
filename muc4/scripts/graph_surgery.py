#!/usr/bin/env python3
"""Graph-surgery simulation: common interface.

A structural-causal-model (SCM) simulator over a learned causal graph:
one predictive model per node, each using ONLY that node's parents in
the graph, plus a do-operator (``intervene``) that clamps a node to a
value, cuts its incoming edges, and propagates through the graph in
topological order.

This module is environment-agnostic: it operates on named variables and
numpy arrays. An environment adapter supplies (1) a graph JSON
(variables + parents per target), (2) training rows of (inputs ->
targets), and (3) optionally a real-environment stepper for verification.
The CartPole adapter is ``run_graph_surgery_cartpole.py``.

MicroRTS STATUS: NOT RUN. No MicroRTS experiment has been executed with
this code. The interface is written so a MicroRTS adapter (graph JSON
from ``muc3/results/microrts/``, replay arrays, env stepper) can plug in
later, but nothing here has been validated on MicroRTS. Do not cite this
module as MicroRTS evidence.

Semantics (read before reusing): the graphs this was built for are
one-step DBNs — causes are variables at time ``t`` and targets are
variables at time ``t+1`` (see ``muc3/scripts/cartpole_bart_target.py``).
``intervene`` therefore clamps a time-``t`` node and recomputes the
time-``t+1`` nodes from their (possibly intervened) parents. With a
one-layer DBN there is no multi-hop propagation to ablate: a
"no-graph" baseline differs ONLY in that each model uses ALL time-``t``
features instead of the learned parents. That is exactly what
``AllFeaturesSurrogate`` implements, so the comparison tests
parent-restriction, not multi-hop propagation. Multi-hop propagation
only becomes testable on a genuinely multi-layer graph.

Caveats (also in the experiment report): the graph is learned from
observational data under untested causal-sufficiency assumptions; BART
edge sets vary by +-1-2 edges across reruns; interventions can fall
outside the joint training distribution even when each value is inside
its observed marginal range; tree models do not extrapolate.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np


class CausalGraph:
    """A DAG over named variables: parents per (non-root) node."""

    def __init__(self, variables: list[str], parents: dict[str, list[str]]):
        self.variables = list(variables)
        self.parents = {node: list(ps) for node, ps in parents.items()}
        unknown = {p for ps in self.parents.values() for p in ps} - set(
            self.variables
        )
        if unknown:
            raise ValueError(f"parents not in variables: {sorted(unknown)}")
        if set(self.parents) - set(self.variables):
            raise ValueError("target nodes must be listed in variables")
        self._topo_order = self._topological_order()

    @classmethod
    def from_json(cls, path: str | Path) -> "CausalGraph":
        """Load a graph JSON with ``variables`` and ``parents`` keys.

        Also accepts the staged CartPole graph format
        (``variables_t`` + ``targets`` with ``target``/``parents``),
        expanding it to the 10-node two-slice DBN (``*_t`` roots and
        ``*_next`` targets). Use :func:`dbn_from_cartpole_json` for that
        format instead; this loader is for the generic format.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["variables"], data["parents"])

    @property
    def targets(self) -> list[str]:
        return [v for v in self.variables if v in self.parents]

    @property
    def roots(self) -> list[str]:
        return [v for v in self.variables if v not in self.parents]

    def edges(self) -> list[tuple[str, str]]:
        return [(p, t) for t in self.targets for p in self.parents[t]]

    def _topological_order(self) -> list[str]:
        children: dict[str, list[str]] = {v: [] for v in self.variables}
        indegree = {v: 0 for v in self.variables}
        for target, ps in self.parents.items():
            for p in ps:
                children[p].append(target)
                indegree[target] += 1
        queue: deque[str] = deque(
            [v for v in self.variables if indegree[v] == 0]
        )
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.variables):
            raise ValueError("graph has a cycle; topological order impossible")
        return order

    @property
    def topo_order(self) -> list[str]:
        return list(self._topo_order)


def dbn_from_cartpole_json(path: str | Path) -> CausalGraph:
    """Expand the staged CartPole graph JSON into a two-slice DBN.

    Time-``t`` variables become roots (``<name>``); each learned target
    becomes a time-``t+1`` node (``<name>_next``) whose parents are the
    time-``t`` variables listed in the graph file.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    variables = list(data["variables_t"]) + [
        t["target_next"] for t in data["targets"]
    ]
    parents = {t["target_next"]: list(t["parents"]) for t in data["targets"]}
    return CausalGraph(variables, parents)


class PerNodeSurrogate:
    """One sklearn model per target node, using only its graph parents.

    ``fit`` takes a dict ``{variable: 1D array}`` covering every variable
    in the graph (roots and targets). ``predict`` evaluates the targets
    from a dict of root values. ``intervene`` applies do-operations:
    clamped nodes keep their assigned value and their incoming edges
    are ignored (their models are never evaluated for them); all other
    nodes are evaluated in topological order from the (possibly
    intervened) values of their parents.
    """

    def __init__(self, graph: CausalGraph, model_factory, classifiers=()):
        """``model_factory``: callable(node_name) -> unfitted estimator.

        ``classifiers``: node names whose estimator is a classifier
        (predicted values are returned as-is, no float coercion beyond
        what the estimator returns).
        """
        self.graph = graph
        self.model_factory = model_factory
        self.classifiers = set(classifiers)
        self.models: dict[str, object] = {}

    def fit(self, columns: dict[str, np.ndarray]) -> "PerNodeSurrogate":
        for target in self.graph.targets:
            parents = self.graph.parents[target]
            if not parents:
                raise ValueError(
                    f"target {target!r} has no parents; constant/empty "
                    "targets need an explicit model (not implemented)"
                )
            X = np.column_stack([np.asarray(columns[p]) for p in parents])
            y = np.asarray(columns[target])
            model = self.model_factory(target)
            model.fit(X, y)
            self.models[target] = model
        return self

    def predict_values(
        self, values: dict[str, float], interventions: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Evaluate all nodes. ``values`` must cover every root.

        ``interventions`` maps node -> clamped value (do-operator). A
        clamped node keeps its value; downstream nodes see the clamped
        value through their parent lists.
        """
        interventions = interventions or {}
        result: dict[str, float] = {}
        for node in self.graph.topo_order:
            if node in interventions:
                # do(node = value): cut incoming edges, fix the value.
                result[node] = float(interventions[node])
            elif node in self.models:
                parents = self.graph.parents[node]
                X = np.array(
                    [[result[p] for p in parents]], dtype=float
                )
                result[node] = self.models[node].predict(X)[0]
            elif node in values:
                result[node] = float(values[node])
            else:
                raise ValueError(f"no value, model, or intervention for {node!r}")
        return result

    def predict_targets(
        self, values: dict[str, float], interventions: dict[str, float] | None = None
    ) -> dict[str, float]:
        full = self.predict_values(values, interventions)
        return {t: full[t] for t in self.graph.targets}

    def predict_targets_batch(
        self,
        columns: dict[str, np.ndarray],
        interventions: dict[str, float | np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Vectorized version of :meth:`predict_targets` over many rows.

        ``columns`` maps every ROOT variable to a 1D array of equal
        length. ``interventions`` maps a node to a scalar (same value
        for every row) or a 1D array (one value per row); intervened
        nodes are clamped and their models are not evaluated. Returns
        one 1D array per target. Numerically identical to calling
        :meth:`predict_targets` row by row; much faster because each
        model predicts the whole batch in one call (per-row calls pay
        the estimator's dispatch overhead every time).
        """
        interventions = interventions or {}
        n = len(next(iter(columns.values())))
        result: dict[str, np.ndarray] = {}
        for node in self.graph.topo_order:
            if node in interventions:
                value = interventions[node]
                if np.isscalar(value):
                    result[node] = np.full(n, float(value))
                else:
                    result[node] = np.asarray(value, dtype=float)
            elif node in self.models:
                parents = self.graph.parents[node]
                X = np.column_stack([result[p] for p in parents])
                result[node] = np.asarray(
                    self.models[node].predict(X), dtype=float
                )
            elif node in columns:
                result[node] = np.asarray(columns[node], dtype=float)
            else:
                raise ValueError(f"no column, model, or intervention for {node!r}")
        return {t: result[t] for t in self.graph.targets}


class AllFeaturesSurrogate(PerNodeSurrogate):
    """Baseline: same interface, but every target uses ALL root features.

    This is the no-graph-propagation baseline: edit a feature, then
    predict directly with a model that sees everything. No edge is cut
    and no sparsity is imposed. It reuses ``intervene`` mechanics only
    in the sense that clamped time-``t`` values are fed in as ordinary
    features; with a one-layer DBN that is exactly "predict directly".
    """

    def __init__(self, graph: CausalGraph, model_factory, classifiers=()):
        complete_parents = {
            target: list(graph.roots) for target in graph.targets
        }
        complete = CausalGraph(graph.variables, complete_parents)
        super().__init__(complete, model_factory, classifiers)


def load_columns_for_dbn(
    states_t: np.ndarray,
    actions_t: np.ndarray,
    states_next: np.ndarray,
    actions_next: np.ndarray,
    state_names: list[str],
    action_name: str = "push",
) -> dict[str, np.ndarray]:
    """Build the ``{variable: column}`` dict for a two-slice DBN."""
    columns: dict[str, np.ndarray] = {}
    for j, name in enumerate(state_names):
        columns[name] = np.asarray(states_t)[:, j]
        columns[f"{name}_next"] = np.asarray(states_next)[:, j]
    columns[action_name] = np.asarray(actions_t)
    columns[f"{action_name}_next"] = np.asarray(actions_next)
    return columns
