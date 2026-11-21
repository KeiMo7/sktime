Point-in-time data and vintage-aware backtesting
=================================================

.. currentmodule:: sktime.vintage

Why revisions break ordinary backtests
--------------------------------------

Monthly business indicators are typically first released as preliminary
values and revised one or two months later. If a backtest is run on the final
revised table, every fold implicitly sees numbers that were only published in
the future. Backtest error then looks much smaller than the error observed
after the model goes live, because the live model only ever sees preliminary
values.

The fix is to record, for every observation, not just its final value but
every value that was ever published for it, together with the publication
instant. A backtest fold at point-in-time cutoff ``t`` then uses only values
published at or before ``t``.

The identity triple
-------------------

Every row in a :class:`VintageFrame` is uniquely identified by

* the series identity (one or more index levels),
* the target period the observation refers to,
* the publication instant at which the value became visible.

Rows are never overwritten: a revision is a new row with a later publication
instant. Duplicate identity triples, null identity components, values
published before their target period, and null values are rejected at
construction time, with the offending series and period named in the error
message.

.. code-block:: python

    import pandas as pd
    from sktime.vintage import VintageFrame

    periods = pd.period_range("2024-01", periods=6, freq="M")
    rows = [("s", p, p + 1, float(k)) for k, p in enumerate(periods)]
    # revision of the January value, published in March
    rows.append(("s", periods[0], pd.Period("2024-03", "M"), 99.0))
    idx = pd.MultiIndex.from_tuples(
        [(a, b, c) for a, b, c, _ in rows],
        names=["series", "target_period", "published_at"],
    )
    vf = VintageFrame(pd.DataFrame({"value": [r[3] for r in rows]}, index=idx))

Point-in-time snapshots
-----------------------

:meth:`VintageFrame.as_of` returns the latest version of every observation
that was knowable at a given instant. Periods not yet published are simply
absent; they are never backfilled.

.. code-block:: python

    vf.as_of(pd.Period("2024-02", "M"))  # January still shows its initial value
    vf.as_of(pd.Period("2024-03", "M"))  # January now shows the revision

Vintage-aware backtesting
-------------------------

:class:`VintageAwareSplitter` wraps any ordinary sktime splitter. The wrapped
splitter still defines the fold windows on the final table, but every fold's
training data is replaced by the as-of snapshot at its cutoff, and the
forecaster is cloned per fold so no fitted state or cache can cross origins.
:func:`vintage_evaluate` then runs the same fold evaluation and parallel
backends as :func:`sktime.forecasting.model_evaluation.evaluate`.

.. code-block:: python

    from sktime.forecasting.naive import NaiveForecaster
    from sktime.split import ExpandingWindowSplitter
    from sktime.vintage import VintageAwareSplitter, vintage_evaluate

    cv = VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=3, step_length=1)
    )
    results = vintage_evaluate(NaiveForecaster(strategy="last"), vf, cv)

The results table has the same columns as ``evaluate``, plus an ``as_of``
column recording each fold's point-in-time cutoff.

Scoring targets
---------------

By default folds are scored against the *initial release* of the test
periods - the number the model would actually have been judged against in
real time. ``score_against="latest"`` scores against the revised final values
instead, which is useful for quantifying the effect of revisions;
``score_against="as_of"`` uses the latest version visible at the fold cutoff
and requires the decision point to be shifted via ``lag`` if test targets are
published later.

When no version information exists
-----------------------------------

An ordinary snapshot can be wrapped with :meth:`VintageFrame.from_snapshot`.
Every observation then has exactly one version, visible immediately, and a
vintage backtest is fold-by-fold identical - including the metric values - to
an ordinary ``evaluate`` backtest on the same snapshot. Existing call sites do
not need to change.

Revision diagnostics
--------------------

:func:`revision_report` lists every revised period with its initial and final
value and the absolute and relative change; :func:`revision_summary`
aggregates the statistics per series; :func:`revision_impact` runs the
backtest twice (initial vs. revised scoring targets) and reports the
fold-wise difference in the metrics.

Error handling
--------------

If a fold has no visible training or test data, the splitter either raises
:class:`InsufficientVintageDataError` naming the fold and cutoff, or skips
the fold with an explicit warning when constructed with
``on_insufficient="skip"``. Future data is never used to fill the gap.
