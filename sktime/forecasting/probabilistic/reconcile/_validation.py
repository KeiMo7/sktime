# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Input validation for probabilistic hierarchical reconciliation.

Two validations are mandated by the design contract:

1. hierarchy additivity (structural; handled in ``_structure``)
2. quantile/interval ordering ("no crossing"): at every node and time point,
   a smaller quantile probability must not carry a value above a larger one.

Crossing is never repaired silently. The reconciler reports every offending
location first, and only proceeds with a repair if the caller explicitly
opts in.
"""

__all__ = ["find_quantile_crossings", "quantile_frame_to_array", "array_to_quantile_frame"]

import numpy as np
import pandas as pd

from sktime.forecasting.probabilistic.reconcile._exceptions import (
    QuantileCrossingError,
)


def find_quantile_crossings(quantiles, alphas):
    """Locate every quantile-order violation in a hierarchical quantile frame.

    Parameters
    ----------
    quantiles : np.ndarray
        Array of shape ``(n_time, n_nodes, n_quantiles)`` where the last
        axis is ordered like ``alphas``.
    alphas : list of float
        Quantile probabilities in the same order as the last axis.

    Returns
    -------
    crossings : list of dict
        One dict per violation with keys: ``time_index`` (int),
        ``node_index`` (int), ``alpha_low``, ``alpha_high``,
        ``value_low``, ``value_high``.
    """
    order = np.argsort(np.asarray(alphas))
    sorted_alphas = np.asarray(alphas)[order]
    sorted_q = quantiles[..., order]

    crossings = []
    diffs = np.diff(sorted_q, axis=-1)
    bad = np.argwhere(diffs < -1e-10)
    for t, node, k in bad:
        crossings.append(
            {
                "time_index": int(t),
                "node_index": int(node),
                "alpha_low": float(sorted_alphas[k]),
                "alpha_high": float(sorted_alphas[k + 1]),
                "value_low": float(sorted_q[t, node, k]),
                "value_high": float(sorted_q[t, node, k + 1]),
            }
        )
    return crossings


def raise_on_crossings(crossings, node_index, time_index, var_name):
    """Turn crossing locations into a detailed ``QuantileCrossingError``."""
    if not crossings:
        return
    lines = []
    details = []
    for c in crossings[:50]:
        node = node_index[c["node_index"]]
        t = time_index[c["time_index"]]
        node_s = node if not isinstance(node, tuple) else "/".join(map(str, node))
        lines.append(
            f"variable={var_name}, node={node_s}, time={t}: "
            f"q({c['alpha_low']})={c['value_low']:.6g} > "
            f"q({c['alpha_high']})={c['value_high']:.6g}"
        )
        details.append({**c, "node": node_s, "time": str(t), "variable": var_name})
    extra = "" if len(crossings) <= 50 else f"\n... and {len(crossings) - 50} more."
    raise QuantileCrossingError(
        "Input quantiles are crossed (lower quantile exceeds higher quantile) "
        f"at {len(crossings)} location(s); no silent repair is performed. "
        "Re-run with on_input_crossing='isotonic' to repair explicitly.\n  "
        + "\n  ".join(lines)
        + extra,
        details=details,
    )


def quantile_frame_to_array(frame, node_index, alphas, var_name):
    """Convert an sktime hierarchical quantile frame to a numeric 3-D array.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of ``predict_quantiles``: hierarchical row index,
        ``(variable, alpha)`` column multi-index.
    node_index : pd.MultiIndex
        Canonical node ordering (from ``HierarchyStructure``).
    alphas : list of float
        Requested quantile probabilities.
    var_name : str
        Variable (column level 0) to extract.

    Returns
    -------
    arr : np.ndarray, shape (n_time, n_nodes, n_quantiles)
    time_index : pd.Index
    """
    sub = frame[var_name].sort_index()
    time_index = sub.index.get_level_values(-1).unique()

    # vectorised reshape: the frame is sorted, so pivot to a
    # (node, time) grid and reindex to the canonical node ordering
    wide = sub[alphas].copy()
    node_level = sub.index.droplevel(-1)

    def _as_tuple(idx):
        # a single-level hierarchy yields a scalar node key, not a tuple
        return idx if isinstance(idx, tuple) else (idx,)

    node_keys = node_level.map(_as_tuple)
    time_keys = sub.index.get_level_values(-1)
    wide = wide.assign(_node=node_keys, _time=time_keys)
    grid = wide.pivot(index="_node", columns="_time", values=alphas)
    # grid columns: (alpha, time); ensure canonical node order
    grid = grid.reindex([_as_tuple(n) for n in node_index])
    if grid.isna().any().any():
        raise ValueError(
            "Quantile frame does not cover every node/time/alpha combination "
            "of the hierarchy."
        )

    n_t, n_n, n_q = len(time_index), len(node_index), len(alphas)
    arr = np.empty((n_t, n_n, n_q))
    for k, a in enumerate(alphas):
        block = grid[a].reindex(columns=list(time_index))
        arr[:, :, k] = block.values.T
    return arr, time_index


def array_to_quantile_frame(arr, frame_template, node_index, time_index, alphas):
    """Inverse of ``quantile_frame_to_array``; preserves the input layout."""
    var_names = frame_template.columns.get_level_values(0).unique()
    n_var = len(var_names)
    columns = pd.MultiIndex.from_product(
        [list(var_names), alphas], names=frame_template.columns.names
    )
    # build one block per variable; arr is per-variable, stacked on axis=3 if needed
    if arr.ndim == 3:
        blocks = [arr]
    else:
        blocks = [arr[..., v] for v in range(n_var)]

    # node_index may itself be a MultiIndex; build the (node..., time)
    # tuples explicitly instead of a cartesian product of the node levels
    if isinstance(node_index, pd.MultiIndex):
        node_tuples = list(map(tuple, node_index))
    else:
        node_tuples = [(n,) for n in node_index]
    row_tuples = [
        (*node, t) for t in time_index for node in node_tuples
    ]
    rows = pd.MultiIndex.from_tuples(
        row_tuples, names=frame_template.index.names
    )
    n_nodes = len(node_index)
    n_times = len(time_index)
    n_q = len(alphas)

    # arr layout is (time, node, quantile); rows above iterate time first
    data = np.column_stack(
        [
            blocks[v].reshape(n_nodes * n_times, n_q)
            for v in range(n_var)
        ]
    )
    out = pd.DataFrame(data, index=rows, columns=columns, dtype=float)
    return out.sort_index()
