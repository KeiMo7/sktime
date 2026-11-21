#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for VintageFrame: identity, validation, as-of semantics, edge cases."""

__author__ = ["sktime developers"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.vintage import (
    VintageFrame,
    VintageValidationError,
)


# ---------------------------------------------------------------------- #
# fixtures
# ---------------------------------------------------------------------- #
@pytest.fixture
def simple_frame():
    """Single series, periods 2024-01..06, T+1 publication, one revision."""
    periods = pd.period_range("2024-01", periods=6, freq="M")
    rows = [
        ("s", p, p + 1, float(k)) for k, p in enumerate(periods)
    ]
    # revise the first period one publication round later
    rows.append(("s", periods[0], pd.Period("2024-03", "M"), 99.0))
    idx = pd.MultiIndex.from_tuples(
        [(a, b, c) for a, b, c, _ in rows],
        names=["series", "target_period", "published_at"],
    )
    return VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    )


@pytest.fixture
def timestamp_frame():
    """Period targets with timestamp publications (mixed index kinds)."""
    periods = pd.period_range("2024-01", periods=5, freq="M")
    rows = [
        ("s", p, (p + 1).to_timestamp(how="start") + pd.Timedelta(days=14),
         float(k))
        for k, p in enumerate(periods)
    ]
    rows.append(("s", periods[0], pd.Timestamp("2024-03-15"), 99.0))
    idx = pd.MultiIndex.from_tuples(
        [(a, b, c) for a, b, c, _ in rows],
        names=["series", "target_period", "published_at"],
    )
    return VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    )


# ---------------------------------------------------------------------- #
# construction and identity
# ---------------------------------------------------------------------- #
def test_identity_triple_keeps_all_versions(simple_frame):
    # both versions of 2024-01 coexist; the revision did not overwrite
    versions = simple_frame.versions_of("s", pd.Period("2024-01", "M"))
    assert len(versions) == 2
    assert list(versions["value"]) == [0.0, 99.0]
    assert list(versions.index.get_level_values(-1)) == [
        pd.Period("2024-02", "M"),
        pd.Period("2024-03", "M"),
    ]


def test_duplicate_identity_triple_rejected():
    idx = pd.MultiIndex.from_tuples(
        [
            ("s", pd.Period("2024-01", "M"), pd.Period("2024-02", "M")),
            ("s", pd.Period("2024-01", "M"), pd.Period("2024-02", "M")),
        ],
        names=["series", "target_period", "published_at"],
    )
    with pytest.raises(VintageValidationError, match="duplicate publication"):
        VintageFrame(pd.DataFrame({"value": [1.0, 2.0]}, index=idx))


def test_publication_before_period_rejected():
    idx = pd.MultiIndex.from_tuples(
        [("s", pd.Period("2024-02", "M"), pd.Period("2024-01", "M"))],
        names=["series", "target_period", "published_at"],
    )
    with pytest.raises(
        VintageValidationError, match="earlier than the target period"
    ):
        VintageFrame(pd.DataFrame({"value": [1.0]}, index=idx))


def test_null_value_and_null_identity_rejected():
    idx = pd.MultiIndex.from_tuples(
        [("s", pd.Period("2024-01", "M"), pd.Period("2024-02", "M"))],
        names=["series", "target_period", "published_at"],
    )
    with pytest.raises(VintageValidationError, match="null values"):
        VintageFrame(pd.DataFrame({"value": [np.nan]}, index=idx))

    idx_nan = pd.MultiIndex.from_tuples(
        [("s", pd.NaT, pd.Period("2024-02", "M"))],
        names=["series", "target_period", "published_at"],
    )
    with pytest.raises(VintageValidationError, match="null values"):
        VintageFrame(pd.DataFrame({"value": [1.0]}, index=idx_nan))


def test_incomparable_index_types_rejected():
    idx = pd.MultiIndex.from_tuples(
        [("s", pd.Period("2024-01", "M"), "not-a-date")],
        names=["series", "target_period", "published_at"],
    )
    with pytest.raises(VintageValidationError, match="incomparable types"):
        VintageFrame(pd.DataFrame({"value": [1.0]}, index=idx))


def test_append_version_refuses_overwrite(simple_frame):
    with pytest.raises(VintageValidationError, match="already exists"):
        simple_frame.append_version(
            "s",
            pd.Period("2024-01", "M"),
            pd.Period("2024-02", "M"),
            3.0,
        )
    extended = simple_frame.append_version(
        "s",
        pd.Period("2024-01", "M"),
        pd.Period("2024-04", "M"),
        3.0,
    )
    assert len(extended) == len(simple_frame) + 1
    # original frame is untouched
    assert len(simple_frame.versions_of("s", pd.Period("2024-01", "M"))) == 2


# ---------------------------------------------------------------------- #
# as-of point-in-time semantics
# ---------------------------------------------------------------------- #
def test_as_of_picks_latest_visible_version(simple_frame):
    # at 2024-02 only the initial release of Jan is visible
    snap = simple_frame.as_of(pd.Period("2024-02", "M"))
    assert snap.loc[("s", pd.Period("2024-01", "M")), "value"] == 0.0
    # at 2024-03 the revision becomes visible
    snap2 = simple_frame.as_of(pd.Period("2024-03", "M"))
    assert snap2.loc[("s", pd.Period("2024-01", "M")), "value"] == 99.0


def test_as_of_never_backfills_future(simple_frame):
    snap = simple_frame.as_of(pd.Period("2024-02", "M"))
    # only 2024-01 exists by then
    assert list(snap.index.get_level_values(-1)) == [pd.Period("2024-01", "M")]
    empty = simple_frame.as_of(pd.Period("2023-12", "M"))
    assert len(empty) == 0


def test_as_of_mixed_period_and_timestamp(timestamp_frame):
    # period cutoff 2024-03 -> start of March; mid-March revision invisible
    snap = timestamp_frame.as_of(pd.Period("2024-03", "M"))
    assert snap.loc[("s", pd.Period("2024-01", "M")), "value"] == 0.0
    # cutoff 2024-04 -> mid-March revision now visible
    snap2 = timestamp_frame.as_of(pd.Period("2024-04", "M"))
    assert snap2.loc[("s", pd.Period("2024-01", "M")), "value"] == 99.0
    # explicit timestamp cutoff works too
    snap3 = timestamp_frame.as_of(pd.Timestamp("2024-03-15"))
    assert snap3.loc[("s", pd.Period("2024-01", "M")), "value"] == 99.0


def test_initial_release_and_latest(simple_frame):
    initial = simple_frame.initial_release()
    assert initial.loc[("s", pd.Period("2024-01", "M")), "value"] == 0.0
    latest = simple_frame.latest()
    assert latest.loc[("s", pd.Period("2024-01", "M")), "value"] == 99.0
    # unrevised periods are identical in both
    p = pd.Period("2024-02", "M")
    assert initial.loc[("s", p), "value"] == latest.loc[("s", p), "value"]


# ---------------------------------------------------------------------- #
# publication calendar, gaps, version-chain diagnostics
# ---------------------------------------------------------------------- #
def test_publication_calendar(simple_frame):
    cal = simple_frame.publication_calendar()
    jan = cal.loc[("s", pd.Period("2024-01", "M"))]
    assert jan["n_versions"] == 2
    assert jan["first_published_at"] == pd.Period("2024-02", "M")
    assert jan["publication_lag"] == 1


def test_find_unpublished():
    periods = pd.period_range("2024-01", periods=5, freq="M")
    rows = [
        ("s", p, p + 1, float(k))
        for k, p in enumerate(periods)
        if k != 2  # 2024-03 never published
    ]
    idx = pd.MultiIndex.from_tuples(
        [(a, b, c) for a, b, c, _ in rows],
        names=["series", "target_period", "published_at"],
    )
    vf = VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    )
    # the frame cannot know about periods it never received by itself;
    # callers pass the expected calendar (e.g. an official release schedule)
    missing = vf.find_unpublished(expected_periods=periods)
    assert ("s", pd.Period("2024-03", "M")) in list(missing)
    # without an external calendar, nothing is "missing"
    assert len(vf.find_unpublished()) == 0


def test_version_chain_completeness_check():
    periods = pd.period_range("2024-01", periods=4, freq="M")
    rows = [("s", p, p + 1, float(k)) for k, p in enumerate(periods)]
    # add a second publication for only one (series, period)
    rows.append(("s", periods[0], pd.Period("2024-03", "M"), 9.0))
    idx = pd.MultiIndex.from_tuples(
        [(a, b, c) for a, b, c, _ in rows],
        names=["series", "target_period", "published_at"],
    )
    vf = VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    )
    with pytest.raises(VintageValidationError, match="version chain gaps"):
        vf.check_version_chain_completeness()


# ---------------------------------------------------------------------- #
# from_snapshot adapter and boundary cases
# ---------------------------------------------------------------------- #
def test_from_snapshot_single_series():
    y = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.period_range("2024-01", periods=3, freq="M"),
        name="value",
    )
    vf = VintageFrame.from_snapshot(y)
    assert vf.n_series == 1
    assert vf.n_periods == 3
    # one version per period, immediately visible -> as_of reproduces y
    snap = vf.as_of(pd.Period("2024-03", "M"))
    pd.testing.assert_series_equal(
        snap.droplevel(0)["value"], y, check_names=False
    )


def test_from_snapshot_hierarchical():
    idx = pd.MultiIndex.from_product(
        [["a", "b"], pd.period_range("2024-01", periods=3, freq="M")],
        names=["series", "target_period"],
    )
    y = pd.DataFrame({"value": np.arange(6.0)}, index=idx)
    vf = VintageFrame.from_snapshot(y)
    assert vf.n_series == 2
    assert set(vf.series_ids()) == {"a", "b"}


def test_from_records_roundtrip(simple_frame):
    df = simple_frame.data.reset_index()
    rebuilt = VintageFrame.from_records(
        df,
        period_col="target_period",
        published_col="published_at",
        series_cols=["series"],
        value_cols=["value"],
    )
    assert rebuilt == simple_frame


def test_empty_and_single_version_boundaries():
    # single row
    idx = pd.MultiIndex.from_tuples(
        [("s", pd.Period("2024-01", "M"), pd.Period("2024-02", "M"))],
        names=["series", "target_period", "published_at"],
    )
    vf = VintageFrame(pd.DataFrame({"value": [1.0]}, index=idx))
    assert len(vf.as_of(pd.Period("2024-02", "M"))) == 1
    assert len(vf.as_of(pd.Period("2024-01", "M"))) == 0
    # empty revision set is fine: a frame without revisions validates
    assert vf.publication_calendar().iloc[0]["n_versions"] == 1
