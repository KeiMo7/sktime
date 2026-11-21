#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter producing point-in-time correct folds from a ``VintageFrame``.

The :class:`VintageAwareSplitter` wraps any ordinary sktime temporal splitter
and turns its folds into *vintage folds*: the training data of every fold
contains only values whose publication instant predates the fold's
point-in-time cutoff, so revisions published later can never leak into
fit, feature construction, transforms, or evaluation.
"""

__author__ = ["sktime developers"]
__all__ = ["VintageAwareSplitter"]

import warnings

import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.vintage._exceptions import (
    InsufficientVintageDataError,
    VintageValidationError,
)
from sktime.vintage._frame import VintageFrame


class VintageAwareSplitter(BaseSplitter):
    """Temporal splitter enforcing point-in-time visibility on vintage data.

    The splitter wraps an ordinary sktime temporal splitter (``cv``) and
    reinterprets every fold on a :class:`~sktime.vintage.VintageFrame`:

    1. the latest snapshot of the vintage frame is built and passed to
       ``cv`` to determine the fold's train/test target periods and the
       fold cutoff (last training target period);
    2. the fold's *point-in-time cutoff* ``as_of`` equals the fold cutoff,
       optionally shifted by ``lag`` (see below);
    3. the training fold is the as-of snapshot: a row for target period
       ``p`` is included only if its publication instant is at or before
       ``as_of``; the value retained is the latest version published by then;
    4. the test fold is the scoring version of the test target periods
       (initial release by default).

    Visibility is therefore enforced row by row through the publication
    instants: a revision published after ``as_of`` never appears in the
    training fold, even if it refers to a period long in the past. Fitted
    state and caches cannot cross folds, because every fold materializes its
    own snapshot and the forecaster is cloned per fold.

    Parameters
    ----------
    cv : sktime BaseSplitter
        Temporal splitter defining the fold windows and forecasting horizon,
        e.g., :class:`~sktime.split.ExpandingWindowSplitter`.
    lag : int or pd.DateOffset/timedelta, optional, default=0
        Optional shift between the fold cutoff and the decision instant.

        - ``lag=0`` (default): the fold origin ``as_of`` is the cutoff
          itself; visibility is determined entirely by the rows'
          publication instants;
        - integer: ``as_of`` is the cutoff shifted by ``lag`` steps along
          the frame's period index;
        - timedelta/DateOffset: added to datetime-like cutoffs.

        A positive lag models decisions made ``lag`` periods after the
        cutoff. Note the common "the indicator for period T is published in
        T+1" situation does **not** require setting the lag: the publication
        instants in the frame already ensure the T value is invisible at T.
    on_insufficient : {"raise", "skip"}, default="raise"
        Behaviour when a fold has no visible training or test data:

        - "raise": raise :class:`InsufficientVintageDataError`, naming the
          fold and the as-of cutoff;
        - "skip": emit a warning and skip the fold (it is absent from the
          generator and the results).

        Future data is never used to fill the gap.
    score_against : {"initial", "as_of", "latest"}, default="initial"
        Which published version of the test periods is used for scoring:

        - "initial": the first published value (realistic backtest: this is
          the number the model would have been judged against in real time);
        - "as_of": the latest version visible at the fold's as-of cutoff;
        - "latest": the latest available version, i.e. the revised final
          value (useful for revision-impact comparisons, see
          :func:`~sktime.vintage.revision_impact`).

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.vintage import VintageFrame, VintageAwareSplitter
    >>> idx = pd.MultiIndex.from_tuples(
    ...     [
    ...         ("s", pd.Period("2024-01", "M"), pd.Period("2024-02", "M")),
    ...         ("s", pd.Period("2024-02", "M"), pd.Period("2024-03", "M")),
    ...         ("s", pd.Period("2024-03", "M"), pd.Period("2024-04", "M")),
    ...         ("s", pd.Period("2024-04", "M"), pd.Period("2024-05", "M")),
    ...     ],
    ...     names=["series", "target_period", "published_at"],
    ... )
    >>> vf = VintageFrame(
    ...     pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    ... )
    >>> inner = ExpandingWindowSplitter(fh=1, initial_window=2)
    >>> vcv = VintageAwareSplitter(inner)
    >>> folds = list(vcv.split_series(vf))
    >>> len(folds)
    2
    """

    _tags = {
        "object_type": "splitter",
        "split_hierarchical": True,
        "split_series_uses": "loc",
        "split_type": "temporal",
        # this splitter consumes VintageFrames, not the ordinary sktime series
        # fixtures assumed by the generic splitter test matrix; its contract
        # is tested in sktime.vintage.tests
        "tests:skip_all": True,
        "tests:core": False,
    }

    def __init__(
        self,
        cv,
        lag=0,
        on_insufficient="raise",
        score_against="initial",
    ):
        if on_insufficient not in ("raise", "skip"):
            raise ValueError(
                "on_insufficient must be 'raise' or 'skip', "
                f"got {on_insufficient!r}"
            )
        if score_against not in ("initial", "as_of", "latest"):
            raise ValueError(
                "score_against must be one of 'initial', 'as_of', 'latest', "
                f"got {score_against!r}"
            )
        self.cv = cv
        self.lag = lag
        self.on_insufficient = on_insufficient
        self.score_against = score_against
        super().__init__()

    def __post_init__(self):
        # do not coerce self.fh: the wrapped splitter owns the horizon
        pass

    @property
    def fh(self):
        """Forecasting horizon, delegated to the wrapped splitter."""
        return self.cv.fh

    def _as_of_for_cutoff(self, cutoff, vf):
        lag = self.lag
        if lag == 0:
            return cutoff
        if isinstance(lag, (pd.Timedelta, pd.DateOffset)):
            return cutoff + lag
        # integer-step shift that works for int, Period and datetime indices
        calendar = vf.periods()
        pos = calendar.get_loc(cutoff)
        new_pos = pos + int(lag)
        if new_pos < 0 or new_pos >= len(calendar):
            raise VintageValidationError(
                f"shifting cutoff {cutoff} by lag={lag} falls outside the "
                f"frame's period range [{calendar[0]}, {calendar[-1]}]"
            )
        return calendar[new_pos]

    def split_series(self, y):
        """Yield point-in-time correct ``(train, test, as_of)`` folds.

        Parameters
        ----------
        y : VintageFrame
            The vintage frame to split. Ordinary sktime containers are
            rejected - wrap them first via
            :meth:`VintageFrame.from_snapshot`.

        Yields
        ------
        y_train : pd.DataFrame
            As-of snapshot of the training target periods: only versions
            published at or before the fold's as-of cutoff. For single-series
            frames this has a two-level (series, period) MultiIndex.
        y_test : pd.DataFrame
            Scoring values for the test target periods, per
            ``score_against``.
        as_of : index-like scalar
            The point-in-time cutoff of the fold.
        """
        if not isinstance(y, VintageFrame):
            raise TypeError(
                "VintageAwareSplitter.split_series expects a VintageFrame; "
                f"got {type(y)}. Wrap an ordinary snapshot with "
                "VintageFrame.from_snapshot first."
            )
        vf = y
        snapshot = vf.latest()
        # ordinary sktime splitter determines fold windows on the final table
        inner_folds = list(self.cv.split_series(snapshot))

        for fold_i, (train_snap, test_snap) in enumerate(inner_folds):
            train_periods = _time_level_values(train_snap.index).unique()
            test_periods = _time_level_values(test_snap.index).unique()
            if len(train_periods) == 0:
                self._handle_short(fold_i, as_of=None, reason="empty train window")
                continue
            cutoff = train_periods.max()
            as_of = self._as_of_for_cutoff(cutoff, vf)

            # row-level point-in-time filter: latest version published <= as_of
            visible = vf.as_of(as_of)
            y_train = _loc_periods(visible, train_periods)
            y_test = self._build_test(vf, test_periods, as_of)

            ok = self._check_sufficiency(fold_i, as_of, y_train, y_test)
            if ok:
                yield y_train.sort_index(), y_test.sort_index(), as_of

    def _build_test(self, vf, test_periods, as_of):
        """Materialize the scoring versions of the test target periods."""
        test_periods = pd.Index(test_periods).unique().sort_values()
        if self.score_against == "latest":
            source = vf.latest()
        elif self.score_against == "initial":
            source = vf.initial_release()
        else:  # "as_of"
            source = vf.as_of(as_of)
        return _loc_periods(source, test_periods)

    def _check_sufficiency(self, fold_i, as_of, y_train, y_test):
        problems = []
        if len(y_train) == 0:
            problems.append("no visible training observations")
        if len(y_test) == 0:
            problems.append("no visible test observations")
        if not problems:
            return True
        return self._handle_short(fold_i, as_of, reason="; ".join(problems))

    def _handle_short(self, fold_i, as_of, reason):
        msg = (
            f"fold {fold_i} (as_of={as_of}): {reason}. The fold cannot be "
            "evaluated without future data, which is never used to fill gaps."
        )
        if self.on_insufficient == "raise":
            raise InsufficientVintageDataError(msg)
        warnings.warn(msg + " Skipping fold.", stacklevel=2)
        return False

    # -- BaseSplitter interface: integer splits over the latest snapshot -- #
    def _split(self, y):
        vf = y if isinstance(y, VintageFrame) else VintageFrame.from_snapshot(y)
        snapshot = vf.latest()
        yield from self.cv.split(snapshot)

    def get_n_splits(self, y=None) -> int:
        """Return number of folds (before any insufficient-data skipping)."""
        if y is None:
            return self.cv.get_n_splits()
        snapshot = y.latest() if isinstance(y, VintageFrame) else y
        return self.cv.get_n_splits(snapshot)

    def get_cutoffs(self, y=None) -> np.ndarray:
        """Return fold cutoffs, delegated to the wrapped splitter."""
        if y is None:
            return self.cv.get_cutoffs()
        snapshot = y.latest() if isinstance(y, VintageFrame) else y
        return self.cv.get_cutoffs(snapshot)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the splitter."""
        from sktime.split import ExpandingWindowSplitter

        return [
            {"cv": ExpandingWindowSplitter(fh=1, initial_window=2)},
            {
                "cv": ExpandingWindowSplitter(fh=[1, 2], initial_window=3),
                "lag": 1,
                "score_against": "as_of",
            },
        ]


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _time_level_values(index):
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values(-1)
    return index


def _loc_periods(frame, periods):
    """Subset a snapshot frame to given target-period values."""
    periods = pd.Index(periods).unique()
    if isinstance(frame.index, pd.MultiIndex):
        time_vals = frame.index.get_level_values(-1)
        return frame[time_vals.isin(periods)]
    return frame[frame.index.isin(periods)]
