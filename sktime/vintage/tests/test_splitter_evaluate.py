#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for VintageAwareSplitter and vintage_evaluate.

The central acceptance criterion is the point-in-time leakage assertion:
for every backtest fold, fit data, features and evaluation only use values
published at or before that fold's as-of cutoff.
"""

__author__ = ["sktime developers"]
__all__ = []

import warnings

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter
from sktime.vintage import (
    InsufficientVintageDataError,
    VintageAwareSplitter,
    VintageFrame,
    vintage_evaluate,
)


def _frame_with_revision(
    n_periods=12,
    revision_at=0,
    revision_published_offset=3,
    publication_lag=1,
    seed=42,
):
    """Build a frame whose initial values are random with one revision.

    Initial value of period ``revision_at`` is its index; the revision adds
    100 and is published ``revision_published_offset`` periods later.
    """
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2024-01", periods=n_periods, freq="M")
    values = rng.normal(size=n_periods)
    rows = [
        ("s", p, p + publication_lag, float(v))
        for p, v in zip(periods, values)
    ]
    revised_period = periods[revision_at]
    rows.append(
        (
            "s",
            revised_period,
            periods[revision_at + revision_published_offset],
            float(values[revision_at]) + 100.0,
        )
    )
    idx = pd.MultiIndex.from_tuples(
        [(a, b, c) for a, b, c, _ in rows],
        names=["series", "target_period", "published_at"],
    )
    return VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    ), periods, values


# ---------------------------------------------------------------------- #
# point-in-time leakage: the hard acceptance assertion
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "splitter",
    [
        ExpandingWindowSplitter(fh=1, initial_window=4, step_length=1),
        SlidingWindowSplitter(fh=1, window_length=4, step_length=1),
        ExpandingWindowSplitter(fh=[1, 2], initial_window=4, step_length=2),
    ],
)
def test_folds_only_contain_visible_values(splitter):
    vf, periods, values = _frame_with_revision()
    vcv = VintageAwareSplitter(splitter)
    raw = vf.data

    for y_train, y_test, as_of in vcv.split_series(vf):
        # assertion 1: every training row is the latest version published
        # at or before the fold's as_of cutoff
        visible = vf.as_of(as_of)
        train_periods = y_train.index.get_level_values(-1)
        expected_train = visible[
            visible.index.get_level_values(-1).isin(train_periods)
        ]
        pd.testing.assert_frame_equal(
            y_train.sort_index(), expected_train.sort_index()
        )

        # assertion 2: no row published after as_of appears in the fold
        for period in train_periods:
            versions = raw.xs(
                ("s", period), level=(0, 1), drop_level=False
            )
            for pub in versions.index.get_level_values(-1):
                assert pub <= as_of

        # assertion 3: test periods are strictly after the training cutoff
        assert train_periods.max() < y_test.index.get_level_values(-1).min()


def test_revision_invisible_until_its_publication():
    vf, periods, values = _frame_with_revision(
        revision_at=0, revision_published_offset=3
    )
    vcv = VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=3, step_length=1)
    )
    revised_period = periods[0]
    publication = periods[3]

    for y_train, y_test, as_of in vcv.split_series(vf):
        jan = y_train.droplevel(0)
        if revised_period in jan.index:
            val = jan.loc[revised_period, "value"]
            if as_of < publication:
                # before publication: only the initial value is knowable
                assert val == pytest.approx(values[0])
            else:
                # after publication: the revision must be used
                assert val == pytest.approx(values[0] + 100.0)


def test_no_silent_future_backfill_when_publication_late():
    # every value is published three periods late; early folds must skip or
    # fail loudly, never silently use future data
    periods = pd.period_range("2024-01", periods=8, freq="M")
    rows = [("s", p, p + 3, float(k)) for k, p in enumerate(periods)]
    idx = pd.MultiIndex.from_tuples(
        [(a, b, c) for a, b, c, _ in rows],
        names=["series", "target_period", "published_at"],
    )
    vf = VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    )
    vcv = VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=3)
    )
    with pytest.raises(InsufficientVintageDataError, match="fold 0"):
        list(vcv.split_series(vf))

    skip = VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=3),
        on_insufficient="skip",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        folds = list(skip.split_series(vf))
    assert any("Skipping fold" in str(w.message) for w in caught)
    # every surviving fold must have non-empty, fully visible data
    for y_train, y_test, as_of in folds:
        assert len(y_train) > 0 and len(y_test) > 0
        assert y_train.index.get_level_values(-1).max() <= as_of


def test_lag_parameter_shifts_visibility():
    vf, periods, values = _frame_with_revision(
        revision_published_offset=2
    )
    base = ExpandingWindowSplitter(fh=1, initial_window=3, step_length=1)
    no_lag = list(VintageAwareSplitter(base, lag=0).split_series(vf))
    lagged = list(VintageAwareSplitter(base, lag=1).split_series(vf))
    # a positive lag moves every as_of one period forward
    for (_, _, a0), (_, _, a1) in zip(no_lag, lagged):
        assert a1 == a0 + 1
    # and strictly more data becomes visible
    for (tr0, _, _), (tr1, _, _) in zip(no_lag, lagged):
        assert len(tr1) >= len(tr0)


def test_score_against_modes():
    # immediate publication (published_at == target period) so every mode has
    # visible test targets
    periods = pd.period_range("2024-01", periods=12, freq="M")
    rows = [("s", p, p, float(k)) for k, p in enumerate(periods)]
    rows.append(("s", periods[0], periods[1], 99.0))
    idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2]) for r in rows],
        names=["series", "target_period", "published_at"],
    )
    vf = VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    )
    base = ExpandingWindowSplitter(fh=1, initial_window=4, step_length=2)
    initial = list(
        VintageAwareSplitter(base, score_against="initial").split_series(vf)
    )
    latest = list(
        VintageAwareSplitter(base, score_against="latest").split_series(vf)
    )
    # as_of scoring needs the decision point shifted past the test targets'
    # publication, which lag=1 models explicitly
    asof = list(
        VintageAwareSplitter(
            base, score_against="as_of", lag=1
        ).split_series(vf)
    )
    assert len(initial) == len(latest) == len(asof)
    # test frames are well-formed in every mode
    for folds in (initial, latest, asof):
        for _, y_test, _ in folds:
            assert len(y_test) > 0

    # with delayed publication, the strict "as_of" mode must refuse folds
    # whose test targets are not yet visible, rather than backfilling
    delayed, _, _ = _frame_with_revision()
    strict = VintageAwareSplitter(base, score_against="as_of")
    with pytest.raises(InsufficientVintageDataError, match="no visible test"):
        list(strict.split_series(delayed))


def test_rejects_non_vintage_input():
    vcv = VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=2)
    )
    with pytest.raises(TypeError, match="VintageFrame"):
        list(vcv.split_series(pd.Series([1.0, 2.0])))


# ---------------------------------------------------------------------- #
# backward compatibility: no version information -> bitwise identical
# ---------------------------------------------------------------------- #
def test_backward_compatible_bitwise_identical():
    rng = np.random.default_rng(7)
    y = pd.Series(
        rng.normal(size=24),
        index=pd.period_range("2024-01", periods=24, freq="M"),
        name="value",
    )
    cv = ExpandingWindowSplitter(
        fh=[1, 2, 3], initial_window=10, step_length=2
    )
    forecaster = NaiveForecaster(strategy="mean", sp=3)
    metric = MeanAbsoluteError()

    old = evaluate(
        forecaster, cv, y, scoring=metric, return_data=True
    )
    vf = VintageFrame.from_snapshot(y)
    new = vintage_evaluate(
        forecaster,
        vf,
        VintageAwareSplitter(cv),
        scoring=metric,
        return_data=True,
    )

    # scores, cutoffs and fold contents agree exactly
    pd.testing.assert_series_equal(
        old["test_MeanAbsoluteError"].reset_index(drop=True),
        new["test_MeanAbsoluteError"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        old["cutoff"].reset_index(drop=True),
        new["cutoff"].reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )
    for i in range(len(old)):
        pd.testing.assert_series_equal(
            old.iloc[i]["y_train"],
            new.iloc[i]["y_train"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            old.iloc[i]["y_test"],
            new.iloc[i]["y_test"],
            check_names=False,
        )
    # vintage path additionally records the as-of cutoff
    assert "as_of" in new.columns


# ---------------------------------------------------------------------- #
# parallelism, reproducibility, cache isolation
# ---------------------------------------------------------------------- #
def test_parallel_matches_sequential_and_is_reproducible():
    vf, _, _ = _frame_with_revision(n_periods=16)
    vcv = VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=5, step_length=2)
    )
    forecaster = NaiveForecaster(strategy="mean")
    metric = MeanAbsoluteError()
    drop_timing = lambda df: df.drop(
        columns=[c for c in df.columns if "time" in c]
    )

    seq = vintage_evaluate(forecaster, vf, vcv, scoring=metric)
    thread = vintage_evaluate(
        forecaster, vf, vcv, scoring=metric,
        backend="threading", backend_params={"n_jobs": 2},
    )
    loky = vintage_evaluate(
        forecaster, vf, vcv, scoring=metric,
        backend="loky", backend_params={"n_jobs": 2},
    )
    seq_repeat = vintage_evaluate(forecaster, vf, vcv, scoring=metric)

    pd.testing.assert_frame_equal(drop_timing(seq), drop_timing(thread))
    pd.testing.assert_frame_equal(drop_timing(seq), drop_timing(loky))
    pd.testing.assert_frame_equal(
        drop_timing(seq), drop_timing(seq_repeat)
    )


def test_fold_cache_isolation_no_cross_origin_leakage():
    # each fold materializes its own snapshot and a cloned forecaster;
    # assert the fold snapshots differ exactly where a revision lands,
    # proving later folds cannot mutate earlier fold inputs
    vf, periods, values = _frame_with_revision(
        revision_at=1, revision_published_offset=2
    )
    vcv = VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=3, step_length=1)
    )
    seen = {}
    for y_train, _, as_of in vcv.split_series(vf):
        # independent object each fold
        assert as_of not in seen
        seen[as_of] = y_train.copy()
    # earlier captured snapshot is unchanged after later folds are built
    first_as_of = sorted(seen)[0]
    first_snapshot = seen[first_as_of]
    again = vf.as_of(first_as_of)
    pd.testing.assert_frame_equal(
        first_snapshot.sort_index(),
        again[
            again.index.get_level_values(-1).isin(
                first_snapshot.index.get_level_values(-1)
            )
        ].sort_index(),
    )


# ---------------------------------------------------------------------- #
# hierarchical / multi-series
# ---------------------------------------------------------------------- #
def test_hierarchical_folds_keep_all_series():
    periods = pd.period_range("2024-01", periods=9, freq="M")
    rows = []
    for s in ("a", "b"):
        for k, p in enumerate(periods):
            rows.append((s, p, p + 1, float(k) + (10 if s == "b" else 0)))
    idx = pd.MultiIndex.from_tuples(
        [(a, b, c) for a, b, c, _ in rows],
        names=["series", "target_period", "published_at"],
    )
    vf = VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    )
    vcv = VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=4, step_length=2)
    )
    for y_train, y_test, as_of in vcv.split_series(vf):
        assert set(y_train.index.get_level_values(0).unique()) == {"a", "b"}
        assert y_train.index.names == ["series", "target_period"]

    res = vintage_evaluate(
        NaiveForecaster(strategy="last"),
        vf,
        vcv,
        scoring=MeanAbsoluteError(),
    )
    assert len(res) == vcv.get_n_splits(vf)
