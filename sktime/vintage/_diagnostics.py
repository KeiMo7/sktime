#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Revision diagnostics: which periods were revised, by how much, and impact."""

__author__ = ["sktime developers"]
__all__ = [
    "revision_report",
    "revision_summary",
    "revision_impact",
]

import numpy as np
import pandas as pd


def revision_report(vf, value_column=None, relative=True, eps=1e-12):
    """Report every revised (series, period) and the magnitude of revision.

    A period is *revised* if more than one version was published for it.

    Parameters
    ----------
    vf : VintageFrame
        The vintage frame to diagnose.
    value_column : str, optional
        Which value column to analyse. Required if the frame has more than
        one value column.
    relative : bool, default=True
        Whether to also compute the relative (fractional) change.
    eps : float, default=1e-12
        Small constant guarding division by zero in relative changes.

    Returns
    -------
    report : pd.DataFrame
        One row per (series, period) that has more than one version, with
        columns:

        - ``initial_value``: first published value,
        - ``final_value``: latest published value,
        - ``n_versions``: number of published versions,
        - ``abs_change``: ``final_value - initial_value``,
        - ``rel_change``: absolute change divided by ``|initial_value|``
          (only if ``relative=True``),
        - ``first_published_at`` / ``last_published_at``: publication instants.
    """
    value_column = _resolve_value_column(vf, value_column)
    df = vf.data[[value_column]]
    group_levels = vf.instance_names + [vf.period_name]

    grp = df.groupby(level=group_levels)[value_column]
    initial = grp.first().rename("initial_value")
    final = grp.last().rename("final_value")
    n_versions = grp.size().rename("n_versions")

    pubs = df.groupby(level=group_levels)
    first_pub = pubs.apply(
        lambda g: g.index.get_level_values(-1).min(),
        include_groups=False,
    ).rename("first_published_at")
    last_pub = pubs.apply(
        lambda g: g.index.get_level_values(-1).max(),
        include_groups=False,
    ).rename("last_published_at")

    report = pd.concat([initial, final, n_versions, first_pub, last_pub], axis=1)
    report = report[report["n_versions"] > 1].copy()
    report["abs_change"] = report["final_value"] - report["initial_value"]
    if relative:
        report["rel_change"] = report["abs_change"] / (
            report["initial_value"].abs() + eps
        )
    return report.sort_index()


def revision_summary(vf, value_column=None):
    """Aggregate revision statistics per series.

    Parameters
    ----------
    vf : VintageFrame
        The vintage frame to diagnose.
    value_column : str, optional
        Which value column to analyse; required if there are multiple.

    Returns
    -------
    pd.DataFrame
        One row per series, with counts and magnitudes of revisions:
        total periods, number and fraction revised, mean and maximum absolute
        and relative revision sizes.
    """
    value_column = _resolve_value_column(vf, value_column)
    report = revision_report(vf, value_column=value_column)

    group_levels = vf.instance_names
    # total number of distinct (series, period) identities
    totals = (
        vf.data.groupby(level=group_levels + [vf.period_name])
        .size()
        .groupby(level=group_levels)
        .size()
        .rename("n_periods")
    )

    if report.empty:
        summary = totals.to_frame()
        summary["n_revised"] = 0
        summary["fraction_revised"] = 0.0
        for col in (
            "mean_abs_change",
            "max_abs_change",
            "mean_rel_change",
            "max_rel_change",
        ):
            summary[col] = np.nan
        return summary

    agg = report.groupby(level=group_levels)["abs_change"].agg(
        mean_abs_change=lambda s: s.abs().mean(),
        max_abs_change=lambda s: s.abs().max(),
    )
    if "rel_change" in report.columns:
        agg_rel = report.groupby(level=group_levels)["rel_change"].agg(
            mean_rel_change=lambda s: s.abs().mean(),
            max_rel_change=lambda s: s.abs().max(),
        )
        agg = agg.join(agg_rel)
    n_rev = (
        report.groupby(level=group_levels)
        .size()
        .rename("n_revised")
    )
    summary = totals.to_frame().join(n_rev).fillna({"n_revised": 0})
    summary["n_revised"] = summary["n_revised"].astype(int)
    summary["fraction_revised"] = summary["n_revised"] / summary["n_periods"]
    summary = summary.join(agg)
    return summary


def revision_impact(
    vf,
    forecaster,
    cv,
    scoring=None,
    value_column=None,
    backend=None,
    backend_params=None,
):
    """Quantify how revisions change backtest conclusions.

    Runs the same backtest twice:

    1. against the *initial releases* (what the market saw first), and
    2. against the *latest available* values (the revised final table).

    Both runs use the same fold structure and the same point-in-time visibility
    rule for training data; only the target values used for scoring differ.

    Parameters
    ----------
    vf : VintageFrame
        Vintage frame carrying initial and revised values.
    forecaster : sktime BaseForecaster
        Forecaster to evaluate (cloned per fold).
    cv : VintageAwareSplitter
        Vintage-aware splitter defining the folds and as-of rule.
    scoring : sktime metric or list thereof, optional
        Metric(s) to compute; defaults to symmetric MAPE, like ``evaluate``.
    value_column : str, optional
        Value column to analyse; required if there are multiple.
    backend : str, optional
        Parallelization backend, forwarded to the vintage evaluation.
    backend_params : dict, optional
        Backend parameters, forwarded to the vintage evaluation.

    Returns
    -------
    impact : pd.DataFrame
        Fold-wise results with the scores on initial and latest values and the
        difference (initial minus latest) for every metric.
    """
    from sktime.vintage._evaluate import vintage_evaluate

    value_column = _resolve_value_column(vf, value_column)

    results_initial = vintage_evaluate(
        forecaster=forecaster,
        vf=vf,
        cv=cv,
        scoring=scoring,
        score_against="initial",
        value_column=value_column,
        backend=backend,
        backend_params=backend_params,
    )
    results_latest = vintage_evaluate(
        forecaster=forecaster,
        vf=vf,
        cv=cv,
        scoring=scoring,
        score_against="latest",
        value_column=value_column,
        backend=backend,
        backend_params=backend_params,
    )

    metric_cols_initial = [
        c for c in results_initial.columns if c.startswith("test_")
    ]
    keep = ["cutoff", "as_of", *metric_cols_initial]
    impact = results_initial[keep].merge(
        results_latest[["cutoff", *metric_cols_initial]],
        on="cutoff",
        suffixes=("_initial", "_latest"),
    )
    for col in metric_cols_initial:
        impact[f"{col}_diff"] = (
            impact[f"{col}_initial"] - impact[f"{col}_latest"]
        )
    return impact


def _resolve_value_column(vf, value_column):
    if value_column is not None:
        if value_column not in vf.value_columns:
            raise ValueError(
                f"value_column {value_column!r} not found; available: "
                f"{vf.value_columns}"
            )
        return value_column
    if len(vf.value_columns) == 1:
        return vf.value_columns[0]
    raise ValueError(
        "the frame has multiple value columns "
        f"{vf.value_columns}; value_column must be specified"
    )
