# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the opt-in probabilistic path of ``ReconcilerForecaster``.

The central contract is backward compatibility: with
``reconcile_pred_int`` left at its default (False), point predictions,
estimator tags and return structures are bit-for-bit unchanged.
"""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.probabilistic.reconcile._validation import (
    quantile_frame_to_array,
)
from sktime.forecasting.reconcile import ReconcilerForecaster
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import (
    _bottom_hier_datagen,
    _make_hierarchical,
)

POINT_METHODS = ["bu", "ols", "wls_str", "td_fcst", "mint_shrink"]


def _hier_y():
    agg = Aggregator()
    return agg.fit_transform(
        _bottom_hier_datagen(
            no_bottom_nodes=4, no_levels=2, random_seed=123, length=40
        )
    )


def test_default_keeps_point_predictions_identical():
    """Criterion 10: feature off => point forecasts unchanged exactly."""
    y = _hier_y()
    f = NaiveForecaster(strategy="mean")
    baseline = ReconcilerForecaster(f, method="bu")
    baseline.fit(y)
    before = baseline.predict(fh=[1, 2, 3]).copy()

    # re-instantiate with the new keyword at its default
    again = ReconcilerForecaster(
        f, method="bu", reconcile_pred_int=False
    )
    again.fit(y)
    after = again.predict(fh=[1, 2, 3])
    pd.testing.assert_frame_equal(before, after)
    assert again.get_tag("capability:pred_int") is False


@pytest.mark.parametrize("method", POINT_METHODS)
def test_point_forecast_unchanged_when_enabled(method):
    """Turning the feature on does not alter the point predict path."""
    y = _hier_y()
    f = NaiveForecaster(strategy="mean")

    off = ReconcilerForecaster(f, method=method).fit(y)
    on = ReconcilerForecaster(
        f, method=method, reconcile_pred_int=True
    ).fit(y)
    pd.testing.assert_frame_equal(
        off.predict(fh=[1, 2]), on.predict(fh=[1, 2])
    )


def test_enabled_tag_and_quantile_coherence():
    y = _hier_y()
    f = NaiveForecaster(strategy="mean")
    rec = ReconcilerForecaster(
        f, method="bu", reconcile_pred_int=True
    ).fit(y)
    assert rec.get_tag("capability:pred_int") is True

    q = rec.predict_quantiles(fh=[1, 2], alpha=[0.1, 0.5, 0.9])
    var = q.columns[0][0]
    # non-crossing
    assert np.diff(q[var].values, axis=1).min() >= -1e-9

    arr, _ = quantile_frame_to_array(
        q,
        rec._prob_reconciler_.node_index_,
        [0.1, 0.5, 0.9],
        var,
    )
    S = rec._prob_reconciler_._S
    projector = S @ np.linalg.pinv(S)
    for t in range(arr.shape[0]):
        for k in range(arr.shape[2]):
            resid = arr[t, :, k] - projector @ arr[t, :, k]
            assert np.abs(resid).max() < 1e-8


def test_interval_and_quantile_equivalent():
    y = _hier_y()
    f = NaiveForecaster(strategy="mean")
    rec = ReconcilerForecaster(
        f, method="ols", reconcile_pred_int=True
    ).fit(y)
    q = rec.predict_quantiles(fh=[1], alpha=[0.1, 0.9])
    pi = rec.predict_interval(fh=[1], coverage=[0.8])
    var = q.columns[0][0]
    np.testing.assert_allclose(
        pi[(var, 0.8, "lower")].values, q[(var, 0.1)].values, rtol=1e-12
    )
    np.testing.assert_allclose(
        pi[(var, 0.8, "upper")].values, q[(var, 0.9)].values, rtol=1e-12
    )


def test_return_totals_false_layout_preserved():
    """The no-totals contract of point predict also holds for quantiles."""
    y = _make_hierarchical(
        hierarchy_levels=(2, 3),
        n_columns=1,
        min_timepoints=16,
        max_timepoints=16,
        index_type="period",
    )
    y = Aggregator().fit_transform(y)
    f = NaiveForecaster(strategy="mean")

    rec = ReconcilerForecaster(
        f, method="bu", reconcile_pred_int=True, return_totals=False
    ).fit(y)
    point = rec.predict(fh=[1, 2])
    quant = rec.predict_quantiles(fh=[1, 2], alpha=[0.1, 0.9])
    # identical row index: totals removed in both cases
    pd.testing.assert_index_equal(
        point.index.sort_values(), quant.index.sort_values()
    )
    assert "__total" not in str(quant.index)


def test_single_series_path_unaffected():
    """Criterion (existing behaviour): non-hierarchical input delegates."""
    from sktime.datasets import load_airline

    y = load_airline()
    f = NaiveForecaster(strategy="mean")
    rec = ReconcilerForecaster(
        f, method="bu", reconcile_pred_int=True
    ).fit(y)
    # no hierarchy => the probabilistic reconciler is not fitted and the
    # underlying forecaster answers directly
    assert rec._prob_reconciler_ is None
    q = rec.predict_quantiles(fh=[1, 2], alpha=[0.1, 0.9])
    assert q.index.nlevels == 1


def test_panel_two_level_index_supported():
    """Regression: panel data (single instance level, no __total) works.

    Covers the case where ``droplevel(-1)`` yields scalar node keys rather
    than tuples, which previously raised ``'int' object is not iterable``.
    """
    from sktime.utils._testing.panel import _make_panel

    y = _make_panel(n_instances=4, n_timepoints=14)
    f = NaiveForecaster(strategy="mean")
    rec = ReconcilerForecaster(
        f, method="bu", reconcile_pred_int=True
    ).fit(y)

    quant = rec.predict_quantiles(fh=[1, 2], alpha=[0.1, 0.9])
    interval = rec.predict_interval(fh=[1, 2], coverage=[0.8])
    var = rec.predict_var(fh=[1, 2])

    # Aggregator adds one __total row per time point
    assert len(quant) == (4 + 1) * 2
    assert list(interval.columns.get_level_values(-1).unique()) == [
        "lower",
        "upper",
    ]
    assert not var.isna().any().any()


def test_reproducibility_across_instances():
    y = _hier_y()
    f = NaiveForecaster(strategy="mean")
    a = ReconcilerForecaster(
        f, method="ols", reconcile_pred_int=True
    ).fit(y).predict_quantiles(fh=[1, 2], alpha=[0.1, 0.9])
    b = ReconcilerForecaster(
        f, method="ols", reconcile_pred_int=True
    ).fit(y).predict_quantiles(fh=[1, 2], alpha=[0.1, 0.9])
    pd.testing.assert_frame_equal(a, b)
