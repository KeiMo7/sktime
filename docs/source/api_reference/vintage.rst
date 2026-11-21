.. _vintage_ref:

Point-in-time data and vintage-aware backtesting
=================================================

The :mod:`sktime.vintage` module handles data revisions: indicators that are
first published as preliminary values and revised in later periods.
Backtesting against only the final revised table creates look-ahead bias,
because every fold then "sees" values that were published only later.

The module stores every published version of an observation, keyed by the
identity triple *series identity + target period + publication instant*, and
produces backtest folds that contain only the values knowable at the fold's
point-in-time cutoff.

.. currentmodule:: sktime.vintage

Data container
--------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    VintageFrame

Splitting and evaluation
------------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    VintageAwareSplitter

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    vintage_evaluate

Revision diagnostics
--------------------

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    revision_report
    revision_summary
    revision_impact

Exceptions
----------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    VintageValidationError
    InsufficientVintageDataError
