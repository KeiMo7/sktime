# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Coverage diagnostics for reconciled probabilistic forecasts.

Computes, per hierarchy level and per nominal coverage:

* nominal coverage (the requested level)
* empirical coverage observed on actuals
* the signed deviation (empirical minus nominal)
* mean interval width before and after reconciliation, with ratios

The diagnostics operate on long-format sktime hierarchical frames so they
can be applied to base forecasts and reconciled forecasts identically.
"""

__all__ = [
    "coverage_diagnostics",
    "width_comparison",
]

import numpy as np
import pandas as pd


def _level_of_node(node, level_nodes):
    for lvl, nodes in enumerate(level_nodes):
        if node in nodes:
            return lvl
    return -1


def coverage_diagnostics(
    actuals,
    interval_frame,
    structure,
    coverages,
    reconciled_label="reconciled",
):
    """Compare nominal vs empirical interval coverage by hierarchy level.

    Parameters
    ----------
    actuals : pd.DataFrame
        Realised values with the same hierarchical row index as the
        forecast frame (subset of time points is allowed).
    interval_frame : pd.DataFrame
        Output of ``predict_interval`` (or a frame reshaped to its
        three-level column index: variable, coverage, lower/upper).
    structure : HierarchyStructure
        Fitted hierarchy structure, used to assign nodes to levels.
    coverages : list of float
        Nominal coverages present in the frame's column level 1.
    reconciled_label : str
        Label carried into the output table.

    Returns
    -------
    diag : pd.DataFrame
        One row per (level, coverage) with columns:
        ``source, level, n_nodes, n_timepoints, coverage,
        nominal_coverage, empirical_coverage, coverage_deviation``.
    """
    var_name = interval_frame.columns.get_level_values(0).unique()[0]
    sub = interval_frame[var_name]
    act = actuals.copy()
    if isinstance(act, pd.DataFrame):
        act = act.iloc[:, 0]

    level_nodes = structure.level_nodes_
    rows = []
    for lvl, nodes in enumerate(level_nodes):
        node_set = list(nodes)
        for cov in coverages:
            lower = sub[(cov, "lower")]
            upper = sub[(cov, "upper")]
            hits, count = 0, 0
            for node in node_set:
                lo = lower.loc[lower.index.droplevel(-1).isin([node])]
                hi = upper.loc[upper.index.droplevel(-1).isin([node])]
                a = act.loc[act.index.droplevel(-1).isin([node])]
                common = lo.index.intersection(a.index)
                if len(common) == 0:
                    continue
                inside = (
                    (a.loc[common].values >= lo.loc[common].values)
                    & (a.loc[common].values <= hi.loc[common].values)
                )
                hits += int(inside.sum())
                count += len(common)
            empirical = hits / count if count else np.nan
            rows.append(
                {
                    "source": reconciled_label,
                    "level": lvl,
                    "n_nodes": len(node_set),
                    "n_timepoints": count,
                    "coverage": float(cov),
                    "nominal_coverage": float(cov),
                    "empirical_coverage": empirical,
                    "coverage_deviation": (
                        empirical - float(cov) if count else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def width_comparison(base_frame, reconciled_frame, structure, coverages):
    """Quantify interval width changes per hierarchy level.

    Parameters
    ----------
    base_frame, reconciled_frame : pd.DataFrame
        Interval frames before and after reconciliation, same layout.
    structure : HierarchyStructure
    coverages : list of float

    Returns
    -------
    table : pd.DataFrame
        Columns: ``level, coverage, mean_width_base, mean_width_reconciled,
        width_ratio, width_change_pct``. A ratio of 1.0 means unchanged;
        below 1.0 the reconciled intervals are narrower.
    """
    var_b = base_frame.columns.get_level_values(0).unique()[0]
    var_r = reconciled_frame.columns.get_level_values(0).unique()[0]
    base = base_frame[var_b]
    rec = reconciled_frame[var_r]

    rows = []
    for lvl, nodes in enumerate(structure.level_nodes_):
        node_list = list(nodes)
        mask_b = base.index.droplevel(-1).isin(node_list)
        mask_r = rec.index.droplevel(-1).isin(node_list)
        for cov in coverages:
            w_b = (
                base[(cov, "upper")][mask_b] - base[(cov, "lower")][mask_b]
            ).mean()
            w_r = (
                rec[(cov, "upper")][mask_r] - rec[(cov, "lower")][mask_r]
            ).mean()
            ratio = float(w_r / w_b) if w_b != 0 else np.nan
            rows.append(
                {
                    "level": lvl,
                    "coverage": float(cov),
                    "mean_width_base": float(w_b),
                    "mean_width_reconciled": float(w_r),
                    "width_ratio": ratio,
                    "width_change_pct": (ratio - 1.0) * 100.0,
                }
            )
    return pd.DataFrame(rows)
