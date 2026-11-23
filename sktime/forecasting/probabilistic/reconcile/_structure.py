# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Hierarchy structure utilities for probabilistic reconciliation.

Builds the summation matrix ``S`` (bottom nodes -> all nodes) and the
parent/child adjacency from an sktime-style hierarchical multi-index, where
aggregate levels are marked by the literal string ``"__total"``.

Also validates additivity, optionally against user-supplied reconciliation
weights (edge weights must sum to one at every parent).
"""

__all__ = ["HierarchyStructure"]

import numpy as np
import pandas as pd

from sktime.forecasting.probabilistic.reconcile._exceptions import (
    HierarchyAdditivityError,
)
from sktime.transformations.hierarchical.reconcile._utils import (
    _get_series_for_each_hierarchical_level,
)

_TOTAL = "__total"


def _vectorized_summing_matrix(node_index):
    """Build the summation matrix without Python-level N x M loops.

    A node contributes to a bottom leaf iff every non-total entry of the
    node equals the corresponding entry of the leaf. This is evaluated with
    broadcast numpy comparisons, one level at a time.
    """
    all_nodes = list(node_index)
    bottom_nodes = [n for n in all_nodes if _TOTAL not in n]
    n_levels = node_index.nlevels

    node_arr = np.array(
        [tuple(n) for n in all_nodes], dtype=object
    )  # (N, L)
    leaf_arr = np.array(
        [tuple(n) for n in bottom_nodes], dtype=object
    )  # (M, L)

    S = np.ones((len(all_nodes), len(bottom_nodes)), dtype=float)
    for level in range(n_levels):
        node_vals = node_arr[:, level][:, None]  # (N, 1)
        leaf_vals = leaf_arr[:, level][None, :]  # (1, M)
        is_total = node_vals == _TOTAL
        match = np.equal(node_vals, leaf_vals)
        S *= np.where(is_total, 1.0, match.astype(float))

    index = pd.MultiIndex.from_tuples(all_nodes, names=node_index.names)
    columns = pd.MultiIndex.from_tuples(bottom_nodes, names=node_index.names)
    return (
        pd.DataFrame(S, index=index, columns=columns)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )


class HierarchyStructure:
    """Structural description of a forecasting hierarchy.

    Parameters
    ----------
    node_index : pd.MultiIndex
        The hierarchy part of a hierarchical frame index, i.e.
        ``frame.index.droplevel(-1).unique()``. Aggregate nodes use the
        ``"__total"`` sentinel, following the sktime ``Aggregator`` convention.
    weights : pd.DataFrame, optional (default=None)
        Optional edge weights as a square DataFrame indexed like the node
        index. ``weights.loc[parent, child]`` is the weight with which the
        child's bottom-level contribution enters the parent's aggregation.
        If ``None``, unit (count) weights are assumed.

    Attributes
    ----------
    S_ : pd.DataFrame
        Summation matrix, shape ``(n_nodes, n_bottom)``; rows are all nodes,
        columns the bottom nodes only.
    parent_child_ : pd.DataFrame
        Two-column frame (``"parent"``, ``"child"``) of direct edges.
    level_nodes_ : list[pd.MultiIndex]
        Nodes at each level of the tree, top to bottom.
    """

    def __init__(self, node_index, weights=None):
        node_index = pd.MultiIndex.from_frame(node_index.to_frame())
        self.node_index = node_index
        self.S_ = _vectorized_summing_matrix(node_index)
        self.parent_child_ = self._build_parent_child()
        self.level_nodes_ = _get_series_for_each_hierarchical_level(node_index)
        self._weights_input = weights
        self.weights_ = self._coerce_weights()
        self._validate_additivity()

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _build_parent_child(self):
        """Direct parent/child edges derived from the summation matrix.

        Vectorised over bottom leaves: for each leaf, the nodes on its
        ancestor path (rows of S containing that leaf) are ordered by
        aggregate depth, and consecutive pairs form the edges.
        """
        s = self.S_
        total_count = (s.index.to_frame() == _TOTAL).sum(axis=1).values
        node_list = list(s.index)
        s_vals = s.values  # (n_nodes, n_bottom)

        edge_pairs = set()
        for j in range(s_vals.shape[1]):
            ancestor_rows = np.where(s_vals[:, j] == 1.0)[0]
            # order by number of __total entries, descending (root first)
            ordered = ancestor_rows[np.argsort(-total_count[ancestor_rows])]
            for a, b in zip(ordered[:-1], ordered[1:]):
                edge_pairs.add((node_list[a], node_list[b]))

        edges = sorted(edge_pairs)
        df = pd.DataFrame(edges, columns=["parent", "child"])
        return df.reset_index(drop=True)
    def _coerce_weights(self):
        """Return a full (n_nodes x n_nodes) edge-weight matrix.

        With unit weights (weights=None), off-diagonal edges carry weight
        1.0; the matrix is only used for user-supplied weighted hierarchies.
        """
        n = len(self.S_.index)
        if self._weights_input is None:
            w_arr = np.zeros((n, n))
            node_pos = {node: i for i, node in enumerate(self.S_.index)}
            edges = self.parent_child_[["parent", "child"]].values
            if len(edges) > 0:
                rows = np.array([node_pos[p] for p in edges[:, 0]])
                cols = np.array([node_pos[c] for c in edges[:, 1]])
                w_arr[rows, cols] = 1.0
            return pd.DataFrame(
                w_arr, index=self.S_.index, columns=self.S_.index
            )
        weights = self._weights_input
        if not isinstance(weights, pd.DataFrame):
            raise TypeError(
                "weights must be a pd.DataFrame indexed by the hierarchy "
                f"nodes, but found {type(weights).__name__}."
            )
        missing = self.S_.index.difference(weights.index)
        if len(missing) > 0 or len(weights.columns.difference(self.S_.index)) > 0:
            raise ValueError(
                "weights must be indexed by exactly the hierarchy nodes; "
                f"missing nodes: {list(map(_fmt_node, missing))}"
            )
        w = weights.loc[self.S_.index, self.S_.index].astype(float).copy()
        return w

    def _validate_additivity(self):
        """Reject the hierarchy if child weights do not sum to one.

        For unit weights this is purely structural: every parent must have
        at least one child. For weighted hierarchies, every parent's direct
        child edge weights must sum to one.
        """
        problems = []
        w = self.weights_.values
        index = self.S_.index
        node_pos = {node: i for i, node in enumerate(index)}

        for parent in self.parent_child_["parent"].unique():
            children = self.parent_child_.loc[
                self.parent_child_["parent"] == parent, "child"
            ]
            child_pos = [node_pos[c] for c in children]
            if self._weights_input is None:
                # structural check only; count hierarchies are always additive
                if len(child_pos) == 0:
                    problems.append(
                        {
                            "parent": _fmt_node(parent),
                            "level": self._level_of(parent),
                            "n_children": 0,
                            "weight_sum": 0.0,
                        }
                    )
                continue
            weight_sum = float(w[node_pos[parent], child_pos].sum())
            if not np.isclose(weight_sum, 1.0, atol=1e-10):
                level = self._level_of(parent)
                problems.append(
                    {
                        "parent": _fmt_node(parent),
                        "level": level,
                        "n_children": int(len(children)),
                        "weight_sum": weight_sum,
                    }
                )
        if problems:
            lines = [
                f"level {p['level']} node {p['parent']}: "
                f"{p['n_children']} children, edge weights sum to "
                f"{p['weight_sum']:.6f} (expected 1.0)"
                for p in problems
            ]
            raise HierarchyAdditivityError(
                "Hierarchy is not additive: child weights must sum to one "
                "for every parent.\n  " + "\n  ".join(lines),
                details=problems,
            )

    def _level_of(self, node):
        for lvl, nodes in enumerate(self.level_nodes_):
            if node in nodes:
                return lvl
        return -1

    # ------------------------------------------------------------------ #
    # accessors used by the reconciler
    # ------------------------------------------------------------------ #
    @property
    def n_nodes(self):
        """Number of nodes across all levels."""
        return self.S_.shape[0]

    @property
    def n_bottom(self):
        """Number of bottom-level (leaf) nodes."""
        return self.S_.shape[1]

    @property
    def bottom_nodes(self):
        """Bottom-level node index."""
        return self.S_.columns

    def aggregation_matrix(self):
        """Weighted summation matrix mapping bottom nodes to all nodes.

        With unit weights this equals ``S_``. With custom edge weights,
        entry ``(node, bottom)`` is the product of edge weights along the
        path from ``bottom`` to ``node`` (zero if not a descendant).
        """
        if self._weights_input is None:
            return self.S_.copy()
        s = self.S_.astype(float).copy()
        # bottom nodes weight themselves with 1.0
        for bottom in self.S_.columns:
            for node in self.S_.index:
                if self.S_.loc[node, bottom] == 0:
                    continue
                s.loc[node, bottom] = self._path_weight(node, bottom)
        return s

    def _path_weight(self, node, bottom):
        """Product of edge weights along the bottom -> node path."""
        if node == bottom:
            return 1.0
        w = self.weights_
        current = bottom
        total = 1.0
        parent_map = dict(
            zip(self.parent_child_["child"], self.parent_child_["parent"])
        )
        # walk up the parent chain until we reach ``node``
        while current != node:
            parent = parent_map[current]
            total *= float(w.loc[parent, current])
            current = parent
        return total


def _fmt_node(node):
    """Render a hierarchy node (str or tuple) as a compact readable string."""
    if isinstance(node, tuple):
        return "/".join(map(str, node))
    return str(node)
