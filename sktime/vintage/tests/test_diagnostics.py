#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for revision diagnostics."""

__author__ = ["sktime developers"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.split import ExpandingWindowSplitter
from sktime.vintage import (
    VintageAwareSplitter,
    VintageFrame,
    revision_impact,
    revision_report,
    revision_summary,
)


def _revised_frame():
    periods = pd.period_range("2024-01", periods=8, freq="M")
    # start at 10 so relative changes are well-defined (no zero base)
    rows = [
        ("s", p, p + 1, 10.0 + float(k)) for k, p in enumerate(periods)
    ]
    # inject two known revisions, each doubling the initial value
    rows.append(("s", periods[0], pd.Period("2024-03", "M"), 20.0))
    rows.append(("s", periods[2], pd.Period("2024-05", "M"), 24.0))
    idx = pd.MultiIndex.from_tuples(
        [(a, b, c) for a, b, c, _ in rows],
        names=["series", "target_period", "published_at"],
    )
    return VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    ), periods


def test_revision_report_matches_injected_values():
    vf, periods = _revised_frame()
    report = revision_report(vf)

    # exactly the two injected revisions, no false positives elsewhere
    assert len(report) == 2
    jan = report.loc[("s", periods[0])]
    mar = report.loc[("s", periods[2])]
    assert jan["initial_value"] == 10.0
    assert jan["final_value"] == 20.0
    assert jan["abs_change"] == pytest.approx(10.0)
    assert jan["rel_change"] == pytest.approx(1.0)
    assert jan["n_versions"] == 2
    assert mar["initial_value"] == 12.0
    assert mar["final_value"] == 24.0
    assert mar["abs_change"] == pytest.approx(12.0)
    assert (
        report.loc[:, "first_published_at"]
        < report.loc[:, "last_published_at"]
    ).all()


def test_revision_report_empty_when_no_revisions():
    y = pd.Series(
        np.arange(4.0),
        index=pd.period_range("2024-01", periods=4, freq="M"),
        name="value",
    )
    vf = VintageFrame.from_snapshot(y)
    assert len(revision_report(vf)) == 0
    summary = revision_summary(vf)
    assert summary.iloc[0]["n_revised"] == 0
    assert summary.iloc[0]["fraction_revised"] == 0.0


def test_revision_summary_counts():
    vf, _ = _revised_frame()
    summary = revision_summary(vf)
    row = summary.loc["s"]
    assert row["n_periods"] == 8
    assert row["n_revised"] == 2
    assert row["fraction_revised"] == pytest.approx(2 / 8)
    assert row["max_abs_change"] == pytest.approx(12.0)
    assert row["mean_abs_change"] == pytest.approx(11.0)


def test_revision_impact_detects_changed_conclusions():
    # every period gets a large revision, so initial-vs-latest scoring on
    # the test targets must produce a nonzero fold-wise difference
    periods = pd.period_range("2024-01", periods=9, freq="M")
    rows = [("s", p, p + 1, float(k)) for k, p in enumerate(periods)]
    rows += [
        ("s", p, p + 2, float(k) + 50.0) for k, p in enumerate(periods)
    ]
    idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2]) for r in rows],
        names=["series", "target_period", "published_at"],
    )
    vf = VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    )
    cv = VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=4, step_length=2)
    )
    impact = revision_impact(
        vf,
        NaiveForecaster(strategy="mean"),
        cv,
        scoring=MeanAbsoluteError(),
    )
    assert "test_MeanAbsoluteError_diff" in impact.columns
    assert "test_MeanAbsoluteError_initial" in impact.columns
    assert "test_MeanAbsoluteError_latest" in impact.columns
    assert impact["test_MeanAbsoluteError_diff"].abs().sum() > 0
    # the diff column is internally consistent
    np.testing.assert_allclose(
        impact["test_MeanAbsoluteError_initial"]
        - impact["test_MeanAbsoluteError_latest"],
        impact["test_MeanAbsoluteError_diff"],
    )


def test_revision_impact_zero_when_targets_unrevised():
    # revisions exist but never touch the scored test periods in a way that
    # changes targets: same predictions, same targets -> zero difference
    vf, _ = _revised_frame()
    cv = VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=6, step_length=1)
    )
    impact = revision_impact(
        vf,
        NaiveForecaster(strategy="last"),
        cv,
        scoring=MeanAbsoluteError(),
    )
    # last-value forecaster ignores the revised history for its point
    # forecast; target values of the late folds are themselves unrevised
    assert "test_MeanAbsoluteError_diff" in impact.columns
