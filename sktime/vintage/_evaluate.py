#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Point-in-time correct backtesting on top of vintage data.

``vintage_evaluate`` is the vintage-aware counterpart of
:func:`sktime.forecasting.model_evaluation.evaluate`. It reuses the exact
same fold evaluation function (``_evaluate_window``) and parallelization
machinery as ``evaluate``; the only difference is how the fold data are
produced - by :class:`~sktime.vintage.VintageAwareSplitter`, which guarantees
that every fold sees only values published before its point-in-time cutoff.
"""

__author__ = ["sktime developers"]
__all__ = ["vintage_evaluate"]

import inspect

import numpy as np
import pandas as pd

from sktime.forecasting.model_evaluation._functions import (
    _check_scores,
    _evaluate_window,
)
from sktime.utils.parallel import parallelize
from sktime.vintage._frame import VintageFrame
from sktime.vintage._splitter import VintageAwareSplitter


def vintage_evaluate(
    forecaster,
    vf,
    cv,
    strategy="refit",
    scoring=None,
    return_data=False,
    error_score=np.nan,
    backend=None,
    backend_params=None,
    return_model=False,
    value_column=None,
    score_against=None,
):
    """Backtest a forecaster on vintage data without look-ahead bias.

    Every fold is evaluated on its own as-of snapshot: fit, feature
    construction, transforms, prediction and scoring all use values whose
    publication instant predates the fold's point-in-time cutoff. The forecaster
    is cloned per fold (``strategy="refit"``), so no fitted state, cache or
    intermediate result is shared between folds - in particular, nothing
    computed at a later origin can leak into an earlier one.

    Parameters
    ----------
    forecaster : sktime BaseForecaster
        Forecaster to benchmark.
    vf : VintageFrame
        Vintage frame carrying all published versions.
    cv : VintageAwareSplitter
        Vintage-aware splitter producing the point-in-time folds.
    strategy : {"refit", "update", "no-update_params"}, default="refit"
        Same semantics as in
        :func:`~sktime.forecasting.model_evaluation.evaluate`. Note that
        parallel backends require ``strategy="refit"``, exactly as there.
    scoring : sktime metric or list of metrics, optional
        Metric(s) to compute; defaults to symmetric MAPE.
    return_data : bool, default=False
        Include fold data columns in the result table.
    error_score : "raise" or numeric, default=np.nan
        Score assigned when fit/predict fails.
    backend : str, optional
        Parallelization backend ("loky", "threading", "dask", ...); see
        :func:`~sktime.forecasting.model_evaluation.evaluate`.
    backend_params : dict, optional
        Additional backend parameters.
    return_model : bool, default=False
        Include the fitted forecaster per fold.
    value_column : str, optional
        Value column to use; required if the frame has multiple value
        columns. The selected column is passed to the forecaster as a
        ``pd.Series`` (single series) or one-column ``pd.DataFrame``
        (panel/hierarchical).
    score_against : {"initial", "as_of", "latest"}, optional
        Override the ``score_against`` setting of ``cv`` for this run.

    Returns
    -------
    results : pd.DataFrame
        Same columns as :func:`~sktime.forecasting.model_evaluation.evaluate`,
        plus an ``as_of`` column carrying the fold's point-in-time cutoff.
    """
    if not isinstance(vf, VintageFrame):
        raise TypeError(
            "vintage_evaluate expects a VintageFrame; got "
            f"{type(vf)}. Use VintageFrame.from_snapshot to wrap an ordinary "
            "snapshot."
        )
    if not isinstance(cv, VintageAwareSplitter):
        raise TypeError(
            "cv must be a VintageAwareSplitter; got "
            f"{type(cv).__name__}."
        )
    if score_against is not None:
        cv = _clone_with(cv, score_against=score_against)

    if strategy not in ("refit", "update", "no-update_params"):
        raise ValueError(
            "strategy must be one of 'refit', 'update', 'no-update_params'"
        )

    scoring = _check_scores(scoring)

    single_series = vf.n_series == 1

    def _select_col(container):
        if value_column is None:
            if len(vf.value_columns) == 1:
                col = vf.value_columns[0]
            else:
                raise ValueError(
                    "the frame has multiple value columns "
                    f"{vf.value_columns}; value_column must be specified"
                )
        else:
            col = value_column
        is_hier = isinstance(container.index, pd.MultiIndex) and (
            container.index.nlevels > 2 or not single_series
        )
        if is_hier:
            return container[[col]]
        # single series: drop the series-identity level, hand over a Series
        if isinstance(container.index, pd.MultiIndex):
            container = container.droplevel(list(range(container.index.nlevels - 1)))
        return container[col]

    def fold_stream():
        for i, (y_train, y_test, as_of) in enumerate(cv.split_series(vf)):
            y_train = _select_col(y_train)
            y_test = _select_col(y_test)
            yield i, (y_train, y_test, None, None), as_of

    folds = list(fold_stream())
    if not folds:
        raise ValueError(
            "no evaluable folds were produced; every fold was skipped due to "
            "insufficient visible data. Set on_insufficient='raise' on the "
            "splitter for details."
        )

    cutoff_dtype = str(vf.periods().dtype)
    # fh is intentionally None: point-in-time filtering can shorten the
    # visible training window relative to the geometric fold, so a relative
    # fh from the wrapped splitter would be wrong. _evaluate_window infers an
    # absolute ForecastingHorizon from y_test, which carries the actual test
    # target periods.
    meta = {
        "fh": None,
        "forecaster": forecaster,
        "scoring": scoring,
        "strategy": strategy,
        "return_data": return_data,
        "return_model": return_model,
        "error_score": error_score,
        "cutoff_dtype": cutoff_dtype,
        "global_mode": False,
    }

    not_parallel = strategy in ("update", "no-update_params")
    as_ofs = [item[2] for item in folds]
    fold_inputs = [(item[0], item[1]) for item in folds]

    if not_parallel:
        results = []
        for x in fold_inputs:
            i = x[0]
            if strategy == "update" or (
                strategy == "no-update_params" and i == 0
            ):
                result, forecaster_fitted = _evaluate_window(x, meta)
                meta["forecaster"] = forecaster_fitted
            else:
                result = _evaluate_window(x, meta)
            results.append(result)
    else:
        results = parallelize(
            fun=_evaluate_window,
            iter=fold_inputs,
            meta=meta,
            backend=backend,
            backend_params=backend_params,
        )

    results = pd.concat(results).reset_index(drop=True)
    results["as_of"] = pd.Series(as_ofs, dtype=object)
    # place as_of next to cutoff for readability
    cols = list(results.columns)
    cols.remove("as_of")
    cutoff_pos = cols.index("cutoff") + 1
    cols = cols[:cutoff_pos] + ["as_of"] + cols[cutoff_pos:]
    results = results[cols]
    return results


def _clone_with(cv, **overrides):
    # use only this class's own __init__ parameters; skbase's get_params
    # also exposes nested params of the wrapped splitter (cv__*) which are
    # not valid constructor arguments here
    own = inspect.signature(cv.__class__.__init__).parameters
    params = {name: getattr(cv, name) for name in own if name != "self"}
    params.update(overrides)
    return cv.__class__(**params)
