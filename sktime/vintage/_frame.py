#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Point-in-time (vintage) data containers and validation.

This module implements the data layer for backtesting with data revisions,
also known as *vintage data* or *point-in-time data*.

The identity of an observation is the triple

    (series identity, target period, publication instant)

see :class:`VintageFrame` for the full documentation.
"""

__author__ = ["sktime developers"]
__all__ = ["VintageFrame", "VintageValidationError"]

import collections.abc

import numpy as np
import pandas as pd

from sktime.vintage._exceptions import VintageValidationError

# standard names of the two time levels in the frame's MultiIndex
PERIOD_LEVEL = "target_period"
PUBLISHED_LEVEL = "published_at"
# default name used for the (single) series-identity level
DEFAULT_INSTANCE_LEVEL = "series"


def _coerce_level_to_index(level, name):
    """Coerce one MultiIndex level (array-like / Index) to a pd.Index."""
    if isinstance(level, pd.Index):
        idx = level.copy()
    else:
        idx = pd.Index(level)
    if idx.name is None:
        idx = idx.rename(name)
    return idx


class VintageFrame:
    """Container storing every published version of time series observations.

    A *vintage frame* keeps, for every target period, every value that was
    ever published for that period (initial release plus every revision),
    together with the instant at which the value became visible.

    This makes it possible to answer point-in-time questions such as:
    "which values of the series were knowable at the end of period T?"
    without looking at revisions that were published only later.
    Using such later revisions in backtesting creates look-ahead bias and is
    the typical cause of backtest error being much smaller than live error.

    **Identity triple.** Each row is uniquely identified by the triple
    ``(series identity, target period, publication instant)``:

    * *series identity*: one or more index levels naming the series. For a
      single series, a single level (default name ``"series"``) is used; for
      hierarchical/panel data, there can be multiple levels.
    * *target period*: the period the observation refers to, level name
      ``target_period``.
    * *publication instant*: the point in time at which this value became
      visible, level name ``published_at``.

    Rows are never silently overwritten: appending a revision with a later
    ``published_at`` adds a new row, it does not replace the initial release.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Long-format frame carrying the observations. The index must be a
        ``pd.MultiIndex`` whose last two levels are the target period and the
        publication instant (in this order). Any preceding levels identify the
        series. If a ``pd.Series`` is passed, the value column is named after
        the series, or ``"value"`` if the series has no name.
    value_columns : str or list of str, optional
        Name(s) of the column(s) holding the observed values. If ``None``,
        every non-index column is treated as a value column.
    validate : bool, default=True
        If True, run the full validation in :meth:`validate` on construction.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples(
    ...     [
    ...         ("a", pd.Period("2024-01", "M"), pd.Period("2024-02", "M")),
    ...         ("a", pd.Period("2024-01", "M"), pd.Period("2024-03", "M")),
    ...         ("a", pd.Period("2024-02", "M"), pd.Period("2024-03", "M")),
    ...     ],
    ...     names=["series", "target_period", "published_at"],
    ... )
    >>> vf = VintageFrame(pd.DataFrame({"value": [1.0, 1.1, 2.0]}, index=idx))
    >>> snap = vf.as_of(pd.Period("2024-02", "M"))
    >>> snap.loc[("a", pd.Period("2024-01", "M")), "value"]
    np.float64(1.0)
    """

    def __init__(self, data, value_columns=None, validate=True):
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError(
                "VintageFrame expects a pandas DataFrame or Series in long "
                f"format with a MultiIndex, but got {type(data)}."
            )
        if isinstance(data, pd.Series):
            name = data.name if data.name is not None else "value"
            data = data.to_frame(name=name)

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(
                "VintageFrame requires a MultiIndex whose last two levels are "
                "the target period and the publication instant, with any "
                "preceding levels identifying the series. Got a plain "
                f"{type(data.index).__name__} instead."
            )
        if data.index.nlevels < 2:
            raise ValueError(
                "VintageFrame requires at least two index levels: "
                "target_period and published_at."
            )

        # rename the last two levels to the canonical names, but only if the
        # user did not give explicit names themselves
        names = list(data.index.names)
        if names[-2] is None:
            names[-2] = PERIOD_LEVEL
        if names[-1] is None:
            names[-1] = PUBLISHED_LEVEL
        if names != list(data.index.names):
            data = data.copy()
            data.index = data.index.set_names(names)

        if value_columns is None:
            value_columns = list(data.columns)
        elif isinstance(value_columns, str):
            value_columns = [value_columns]
        missing = [c for c in value_columns if c not in data.columns]
        if missing:
            raise ValueError(
                f"value_columns {missing} not found in data columns "
                f"{list(data.columns)}"
            )

        self._data = data.sort_index()
        self._value_columns = list(value_columns)
        self._period_name = self._data.index.names[-2]
        self._published_name = self._data.index.names[-1]
        self._instance_names = list(self._data.index.names[:-2])

        if validate:
            self.validate()

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_records(
        cls,
        records,
        period_col=PERIOD_LEVEL,
        published_col=PUBLISHED_LEVEL,
        series_cols=None,
        value_cols=None,
    ):
        """Construct a VintageFrame from a flat records table.

        Parameters
        ----------
        records : pd.DataFrame
            Flat table with one row per published version.
        period_col : str, default="target_period"
            Column naming the target period.
        published_col : str, default="published_at"
            Column naming the publication instant.
        series_cols : str or list of str, optional
            Column(s) identifying the series. If ``None`` and no other
            identifying columns remain, a single constant series is assumed.
        value_cols : str or list of str, optional
            Column(s) holding values. If ``None``, every remaining column is
            used.

        Returns
        -------
        VintageFrame
        """
        if not isinstance(records, pd.DataFrame):
            raise TypeError(f"records must be a pd.DataFrame, got {type(records)}")
        df = records.copy()
        for col in (period_col, published_col):
            if col not in df.columns:
                raise ValueError(f"required column {col!r} not found in records")
        if isinstance(series_cols, str):
            series_cols = [series_cols]
        if not series_cols:
            # no series identity supplied: treat the table as one series
            df[DEFAULT_INSTANCE_LEVEL] = DEFAULT_INSTANCE_LEVEL
            series_cols = [DEFAULT_INSTANCE_LEVEL]
        if value_cols is None:
            value_cols = [
                c
                for c in df.columns
                if c not in series_cols + [period_col, published_col]
            ]
        value_cols = _as_list(value_cols)
        if not value_cols:
            raise ValueError(
                "no value columns found after removing index columns; "
                "pass value_cols explicitly if the table layout is unusual"
            )
        index_cols = series_cols + [period_col, published_col]
        df = df.set_index(index_cols)
        return cls(df[value_cols])

    @classmethod
    def from_snapshot(cls, y, published_at=None, series_name=DEFAULT_INSTANCE_LEVEL):
        """Construct a single-vintage frame from an sktime time series.

        The resulting frame carries exactly one version per target period,
        published at ``published_at``. It is the canonical adapter that makes
        an ordinary (final) snapshot usable with the vintage API; backtests on
        such a frame are identical to backtests on the original series.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or sktime panel/hierarchical container
            Time series in an sktime compatible pandas mtype
            (``pd.Series``, ``pd.DataFrame``, ``pd-multiindex``,
            ``pd_multiindex_hier``).
        published_at : index-like, optional
            Publication instant attached to every row. If ``None``, the
            publication instant is set equal to the target period, i.e. every
            observation is treated as visible immediately.
        series_name : str, default="series"
            Name used for the single series-identity level when ``y`` is a
            plain ``pd.Series`` or ``pd.DataFrame``.

        Returns
        -------
        VintageFrame
        """
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            from sktime.datatypes import convert

            y = convert(y, to_type=["pd_multiindex_hier", "pd-multiindex", "pd.DataFrame"])

        if isinstance(y, pd.Series):
            y = y.to_frame(name=y.name if y.name is not None else "value")

        if isinstance(y.index, pd.MultiIndex):
            instance_levels = list(y.index.names[:-1])
            period_idx = y.index.get_level_values(-1)
        else:
            instance_levels = [series_name]
            period_idx = y.index

        if published_at is None:
            published_idx = period_idx
        else:
            if isinstance(published_at, pd.Index):
                published_idx = published_idx
            else:
                published_idx = pd.Index(published_at)
            if len(published_idx) == 1:
                published_idx = published_idx.repeat(len(period_idx))
            elif len(published_idx) != len(period_idx):
                raise ValueError(
                    "published_at must be a scalar or have the same length as y"
                )

        idx = pd.MultiIndex.from_arrays(
            (
                *(
                    [np.full(len(y), series_name)]
                    if not isinstance(y.index, pd.MultiIndex)
                    else [
                        y.index.get_level_values(i)
                        for i in range(y.index.nlevels - 1)
                    ]
                ),
                period_idx,
                published_idx,
            ),
            names=instance_levels + [PERIOD_LEVEL, PUBLISHED_LEVEL],
        )
        df = pd.DataFrame(y.values, index=idx, columns=y.columns)
        return cls(df)

    # ------------------------------------------------------------------ #
    # basic dunder / attributes
    # ------------------------------------------------------------------ #
    @property
    def data(self):
        """Underlying long-format ``pd.DataFrame`` (copy on write)."""
        return self._data

    @property
    def value_columns(self):
        """Columns holding observed values."""
        return list(self._value_columns)

    @property
    def instance_names(self):
        """Index level names identifying the series (all but last two)."""
        return list(self._instance_names)

    @property
    def period_name(self):
        """Name of the target-period index level."""
        return self._period_name

    @property
    def published_name(self):
        """Name of the publication-instant index level."""
        return self._published_name

    @property
    def index(self):
        """MultiIndex of the underlying frame."""
        return self._data.index

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if not isinstance(other, VintageFrame):
            return NotImplemented
        return self._data.equals(other._data)

    def copy(self):
        """Return a deep copy."""
        return VintageFrame(self._data.copy(), value_columns=self._value_columns)

    def __repr__(self):
        return (
            f"VintageFrame(n_series={self.n_series}, "
            f"n_periods={self.n_periods}, n_versions={len(self)})\n"
            f"{self._data.__repr__()}"
        )

    # ------------------------------------------------------------------ #
    # dimensions
    # ------------------------------------------------------------------ #
    @property
    def n_series(self):
        """Number of distinct series identities."""
        if not self._instance_names:
            return 1
        return self._data.groupby(level=self._instance_names).ngroups

    @property
    def n_periods(self):
        """Number of distinct target periods (union across series)."""
        return self._data.index.get_level_values(-2).nunique()

    def series_ids(self):
        """Return the unique series identities as a ``pd.MultiIndex``."""
        if not self._instance_names:
            return pd.Index([DEFAULT_INSTANCE_LEVEL], name=DEFAULT_INSTANCE_LEVEL)
        return self._data.groupby(level=self._instance_names).size().index

    def periods(self):
        """Return the sorted unique target-period index."""
        return pd.Index(
            self._data.index.get_level_values(-2).unique().sort_values()
        )

    def publication_dates(self):
        """Return the sorted unique publication instants."""
        return pd.Index(
            self._data.index.get_level_values(-1).unique().sort_values()
        )

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self):
        """Validate the frame, raising ``VintageValidationError`` on problems.

        Checks performed:

        1. the index is a MultiIndex with at least two levels;
        2. no component of the identity triple is null;
        3. the identity triple is unique (no duplicate publications);
        4. publication instants are never earlier than their target period;
        5. value columns contain no nulls;
        6. target-period and publication-instant levels have consistent types.

        Returns
        -------
        None
            on success.

        Raises
        ------
        VintageValidationError
            with a message identifying the offending series and period,
            whenever that information is available.
        """
        df = self._data
        idx = df.index

        # 2. null identity components
        for lvl in range(idx.nlevels):
            vals = idx.get_level_values(lvl)
            null_mask = pd.isnull(vals)
            if null_mask.any():
                name = idx.names[lvl]
                bad = idx[null_mask][:3]
                raise VintageValidationError(
                    f"index level {name!r} contains null values; "
                    f"first offending rows: {list(bad)}"
                )

        # 3. duplicate identity triples
        duplicates = idx[idx.duplicated(keep=False)]
        if len(duplicates) > 0:
            dup_unique = duplicates.unique()[:5]
            raise VintageValidationError(
                "duplicate publication(s) found: the triple "
                "(series, target_period, published_at) must be unique. "
                f"First offending identities: {list(dup_unique)}. "
                "If two values were published at the same instant, resolve "
                "the conflict before constructing the VintageFrame."
            )

        # 4. publication earlier than target period
        periods = idx.get_level_values(-2)
        published = idx.get_level_values(-1)
        pub_for_compare = _to_comparable(published, periods)
        if pub_for_compare is not None:
            early = np.asarray(pub_for_compare < periods)
            if early.any():
                bad = idx[early][:5]
                raise VintageValidationError(
                    "publication instant is earlier than the target period "
                    "for at least one observation; a value cannot be visible "
                    f"before the period it refers to. First offenders: {list(bad)}"
                )
        else:
            raise VintageValidationError(
                "target-period and publication-instant levels have "
                f"incomparable types: {periods.dtype} vs {published.dtype}"
            )

        # 5. null values
        for col in self._value_columns:
            null_mask = df[col].isna()
            if null_mask.any():
                bad = df.index[null_mask][:5]
                raise VintageValidationError(
                    f"value column {col!r} contains null values; missing "
                    f"observations must be absent as rows. First offenders: "
                    f"{list(bad)}"
                )

        # per-series ordering / version chain check
        self._check_version_chains()

    def _check_version_chains(self):
        """Check that each (series, period) version chain is well ordered."""
        group_levels = self._instance_names + [self._period_name]
        for key, grp in self._data.groupby(level=group_levels):
            pubs = grp.index.get_level_values(-1)
            if not pubs.is_monotonic_increasing:
                raise VintageValidationError(
                    "version chain is not ordered by publication instant "
                    f"for identity {key}; got publication instants {list(pubs)}"
                )

    def check_version_chain_completeness(self, expected_publishers=None):
        """Check that every series has versions at every expected publication.

        Parameters
        ----------
        expected_publishers : index-like, optional
            Publication instants that every (series, period) present in the
            frame should have. If ``None``, the union of all publication
            instants in the frame is used. Note that a period legitimately
            cannot have publications before it exists, so only instants at or
            after the period are considered.

        Raises
        ------
        VintageValidationError
            identifying the series and period with a broken version chain.
        """
        df = self._data
        all_pubs = (
            self.publication_dates()
            if expected_publishers is None
            else pd.Index(expected_publishers)
        )
        group_levels = self._instance_names + [self._period_name]
        gaps = []
        for key, grp in df.groupby(level=group_levels):
            period = key[-1] if isinstance(key, tuple) else key
            comparable_pubs = _to_comparable(all_pubs, pd.Index([period]))
            if comparable_pubs is None:
                raise VintageValidationError(
                    "expected_publishers have dtype "
                    f"{all_pubs.dtype}, incomparable with target-period dtype "
                    f"{df.index.get_level_values(-2).dtype}"
                )
            expected = all_pubs[comparable_pubs >= period]
            actual = grp.index.get_level_values(-1).unique()
            missing = expected.difference(actual)
            if len(missing) > 0:
                gaps.append((key, list(missing[:5])))
        if gaps:
            preview = gaps[:5]
            raise VintageValidationError(
                "version chain gaps detected: at least one (series, period) "
                "is missing publications present elsewhere. First offenders "
                f"(identity, missing publication instants): {preview}"
            )

    # ------------------------------------------------------------------ #
    # point-in-time operations
    # ------------------------------------------------------------------ #
    def as_of(self, time, include_equal=True):
        """Return the knowable snapshot of the data at instant ``time``.

        For every (series, target period), the latest version whose
        publication instant is at or before ``time`` is retained. Periods that
        have no published version by ``time`` are simply absent from the
        result - they are never backfilled with future data.

        ``time`` may be on a different index kind than the publication level,
        e.g. a ``pd.Period`` cutoff against timestamp publications; period
        cutoffs are interpreted as the end of the period for visibility.

        Parameters
        ----------
        time : index-like scalar
            The point-in-time cutoff.
        include_equal : bool, default=True
            If True, publications at exactly ``time`` count as visible.

        Returns
        -------
        snapshot : pd.DataFrame
            Frame indexed by the series-identity level(s) and the target
            period (the publication-instant level is dropped), in an sktime
            compatible pandas mtype. Values are those knowable at ``time``.
        """
        idx = self._data.index
        published = idx.get_level_values(-1)
        cutoff = _coerce_cutoff(time, published)
        if cutoff is None:
            raise VintageValidationError(
                f"as_of cutoff {time!r} (type {type(time).__name__}) cannot be "
                "compared against publication instants of dtype "
                f"{published.dtype}"
            )
        if include_equal:
            mask = published <= cutoff
        else:
            mask = published < cutoff
        visible = self._data[np.asarray(mask)]
        if visible.empty:
            return self._empty_snapshot()
        group_levels = self._instance_names + [self._period_name]
        # last version per identity = latest publication visible at `time`
        snapshot = visible.groupby(level=group_levels).tail(1)
        snapshot = snapshot.droplevel(-1).sort_index()
        return snapshot

    def _empty_snapshot(self):
        levels = self._instance_names + [self._period_name]
        idx = pd.MultiIndex.from_arrays(
            [pd.Index([], name=n) for n in levels]
        )
        return pd.DataFrame(
            {c: pd.Series(dtype=self._data[c].dtype) for c in self._value_columns},
            index=idx,
        )

    def latest(self):
        """Return the latest available version of every observation."""
        last_pub = self.publication_dates()[-1]
        return self.as_of(last_pub)

    def initial_release(self):
        """Return the first published (initial) version of every observation."""
        group_levels = self._instance_names + [self._period_name]
        snap = self._data.groupby(level=group_levels).head(1)
        return snap.droplevel(-1).sort_index()

    def versions_of(self, series_id, period):
        """Return all published versions for one (series, period).

        Parameters
        ----------
        series_id : tuple or scalar
            Series identity; scalar for a single instance level, tuple
            otherwise.
        period : index-like scalar
            Target period.

        Returns
        -------
        pd.DataFrame
            Rows ordered by publication instant.
        """
        key = _as_tuple(series_id) + (period,)
        return self._data.loc[key].sort_index()

    def publication_calendar(self):
        """Return, per series, the publication instant of every target period.

        Returns
        -------
        pd.DataFrame
            Indexed by (series identity, target period), with columns

            - ``first_published_at``: first publication instant,
            - ``last_published_at``: latest publication instant,
            - ``n_versions``: number of published versions,
            - ``publication_lag``: difference between first publication and
              the target period (in the units of the index).
        """
        group_levels = self._instance_names + [self._period_name]
        grp = self._data.groupby(level=group_levels)
        pubs = grp.size().rename("n_versions").to_frame()
        pubs["first_published_at"] = grp.apply(
            lambda g: g.index.get_level_values(-1).min(), include_groups=False
        )
        pubs["last_published_at"] = grp.apply(
            lambda g: g.index.get_level_values(-1).max(), include_groups=False
        )
        periods = pubs.index.get_level_values(-1)
        first_pubs = pd.Index(pubs["first_published_at"].values)
        coerced = _to_comparable(first_pubs, periods)
        if coerced is None:
            pubs["publication_lag"] = pd.NA
        elif isinstance(periods, pd.PeriodIndex) and isinstance(
            coerced, pd.PeriodIndex
        ):
            # period difference as an integer number of periods
            pubs["publication_lag"] = coerced.asi8 - periods.asi8
        else:
            pubs["publication_lag"] = coerced - periods
        return pubs

    def periods_visible_at(self, time, series_id=None):
        """Return target periods having at least one version visible at ``time``."""
        snap = self.as_of(time)
        if series_id is not None and self._instance_names:
            key = _as_tuple(series_id)
            snap = snap.loc[key]
        return pd.Index(snap.index.get_level_values(-1).unique().sort_values())

    def append_version(self, series_id, period, published_at, values):
        """Return a new ``VintageFrame`` with one additional published version.

        The original frame is not modified. Duplicate identity triples raise
        ``VintageValidationError`` instead of overwriting.

        Parameters
        ----------
        series_id : tuple or scalar
            Series identity.
        period : index-like scalar
            Target period.
        published_at : index-like scalar
            Publication instant.
        values : dict or scalar
            Mapping ``value_column -> value``, or a scalar if there is exactly
            one value column.

        Returns
        -------
        VintageFrame
        """
        key = _as_tuple(series_id) + (period, published_at)
        if key in self._data.index:
            raise VintageValidationError(
                f"a version with identity {key} already exists; revisions must "
                "carry a new publication instant, they cannot overwrite."
            )
        if np.isscalar(values):
            if len(self._value_columns) != 1:
                raise ValueError(
                    "scalar values can only be used when there is exactly one "
                    f"value column; found {self._value_columns}"
                )
            values = {self._value_columns[0]: values}
        unknown = set(values) - set(self._value_columns)
        if unknown:
            raise ValueError(f"unknown value columns: {unknown}")
        row = pd.DataFrame([values], columns=self._value_columns)
        row.index = pd.MultiIndex.from_tuples(
            [key], names=self._data.index.names
        )
        new_data = pd.concat([self._data, row])
        return VintageFrame(new_data, value_columns=self._value_columns)

    # ------------------------------------------------------------------ #
    # missingness diagnostics
    # ------------------------------------------------------------------ #
    def find_unpublished(self, expected_periods=None, series_ids=None):
        """Find (series, period) pairs that were never published.

        Parameters
        ----------
        expected_periods : index-like, optional
            Target periods every series should have. If ``None``, the full
            period range of the frame is used.
        series_ids : index-like, optional
            Series to check; defaults to all series.

        Returns
        -------
        pd.MultiIndex
            Missing (series identity, target period) pairs.
        """
        if expected_periods is None:
            expected_periods = self.periods()
        if series_ids is None:
            series_ids = self.series_ids()
        expected_idx = pd.MultiIndex.from_product(
            [series_ids, expected_periods],
            names=self._instance_names + [self._period_name],
        )
        present_idx = self.latest().index
        return expected_idx.difference(present_idx)


def _to_comparable(published, periods):
    """Coerce publication index so it can be compared with target periods.

    Returns an index comparable to ``periods``, or ``None`` if the two
    index kinds cannot be reconciled.
    """
    if isinstance(periods, pd.PeriodIndex) and isinstance(
        published, pd.DatetimeIndex
    ):
        return published.to_period(freq=periods.freq)
    try:
        pd.Index(published[:1]) <= pd.Index(periods[:1])
        return published
    except TypeError:
        return None


def _coerce_cutoff(time, published):
    """Coerce an as-of cutoff scalar to the publication index's kind.

    Period cutoffs against timestamp publications are interpreted as the
    period's end instant. Returns ``None`` if coercion is impossible.
    """
    time_idx = pd.Index([time])
    if isinstance(published, pd.PeriodIndex):
        if isinstance(time_idx, pd.PeriodIndex):
            if time_idx.freq != published.freq:
                time_idx = time_idx.asfreq(published.freq, how="end")
            return time_idx[0]
        if isinstance(time_idx, pd.DatetimeIndex):
            return time_idx.to_period(freq=published.freq)[0]
    if isinstance(published, pd.DatetimeIndex) and isinstance(
        time_idx, pd.PeriodIndex
    ):
        # a period cutoff represents a decision made within that period:
        # only publications up to the period's start instant are visible.
        # If a publication arrives later in the same period, model the delay
        # explicitly via the splitter's lag parameter.
        return time_idx.to_timestamp(how="start")[0]
    try:
        pd.Index(published[:1]) <= time_idx
        return time
    except TypeError:
        return None


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    if isinstance(x, collections.abc.Iterable):
        return list(x)
    return [x]


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)
