# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Export reconciled forecasts into fixed-schema reconciliation tables.

The output schema is frozen so downstream accounting jobs can diff runs
column-by-column. Column order and level ordering never depend on dict
hashing or dict insertion order of user code.
"""

__all__ = ["export_interval_table", "export_quantile_table"]

import numpy as np
import pandas as pd

# frozen schema - do not reorder without a deprecation cycle
INTERVAL_TABLE_COLUMNS = [
    "level",
    "node",
    "timepoint",
    "variable",
    "coverage",
    "lower",
    "upper",
    "point_forecast",
    "width",
]

QUANTILE_TABLE_COLUMNS = [
    "level",
    "node",
    "timepoint",
    "variable",
    "alpha",
    "quantile",
]


def _node_level_map(structure):
    mapping = {}
    for lvl, nodes in enumerate(structure.level_nodes_):
        for node in nodes:
            mapping[node] = lvl
    return mapping


def _node_label(node):
    if isinstance(node, tuple):
        return "/".join(map(str, node))
    return str(node)


def export_interval_table(
    interval_frame,
    structure,
    point_frame=None,
):
    """Flatten an interval frame into the fixed reconciliation table schema.

    Parameters
    ----------
    interval_frame : pd.DataFrame
        Reconciled interval frame (variable, coverage, lower/upper columns).
    structure : HierarchyStructure
    point_frame : pd.DataFrame, optional
        Reconciled point forecast, merged into the table.

    Returns
    -------
    table : pd.DataFrame
        Columns exactly equal to ``INTERVAL_TABLE_COLUMNS``, sorted by
        level (top -> bottom), node, timepoint, coverage. The sort order is
        deterministic for identical inputs.
    """
    var_name = interval_frame.columns.get_level_values(0).unique()[0]
    sub = interval_frame[var_name]
    level_map = _node_level_map(structure)

    point_lookup = None
    if point_frame is not None:
        point_lookup = point_frame.iloc[:, 0]

    records = []
    coverages = list(sub.columns.get_level_values(0).unique())
    n_node_levels = sub.index.nlevels - 1
    for idx, row in sub.iterrows():
        node_parts = tuple(idx[:n_node_levels])
        timepoint = idx[-1]
        node = node_parts if len(node_parts) > 1 else node_parts[0]
        level = level_map.get(node, -1)
        label = _node_label(node)
        point_val = (
            float(point_lookup.loc[idx])
            if point_lookup is not None and idx in point_lookup.index
            else np.nan
        )
        for cov in coverages:
            lower = float(row[(cov, "lower")])
            upper = float(row[(cov, "upper")])
            records.append(
                {
                    "level": level,
                    "node": label,
                    "timepoint": timepoint,
                    "variable": var_name,
                    "coverage": float(cov),
                    "lower": lower,
                    "upper": upper,
                    "point_forecast": point_val,
                    "width": upper - lower,
                }
            )
    table = pd.DataFrame(records, columns=INTERVAL_TABLE_COLUMNS)
    return table.sort_values(
        ["level", "node", "timepoint", "coverage"]
    ).reset_index(drop=True)


def export_quantile_table(quantile_frame, structure):
    """Flatten a quantile frame into the fixed quantile table schema."""
    var_name = quantile_frame.columns.get_level_values(0).unique()[0]
    sub = quantile_frame[var_name]
    level_map = _node_level_map(structure)
    alphas = sorted(float(a) for a in sub.columns.unique())

    records = []
    n_node_levels = sub.index.nlevels - 1
    for idx, row in sub.iterrows():
        node_parts = tuple(idx[:n_node_levels])
        timepoint = idx[-1]
        node = node_parts if len(node_parts) > 1 else node_parts[0]
        level = level_map.get(node, -1)
        label = _node_label(node)
        for alpha in alphas:
            records.append(
                {
                    "level": level,
                    "node": label,
                    "timepoint": timepoint,
                    "variable": var_name,
                    "alpha": alpha,
                    "quantile": float(row[alpha]),
                }
            )
    table = pd.DataFrame(records, columns=QUANTILE_TABLE_COLUMNS)
    return table.sort_values(
        ["level", "node", "timepoint", "alpha"]
    ).reset_index(drop=True)
