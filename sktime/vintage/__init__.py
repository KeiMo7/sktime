#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Point-in-time (vintage) data handling and leakage-free backtesting.

This module addresses backtesting with *data revisions*: economic and business
indicators are typically first published as preliminary values and revised one
or more periods later. Backtesting against the final revised table creates
look-ahead bias, because every fold then "sees" numbers that were only
published later.

The module provides:

* :class:`VintageFrame` - container storing every published version of every
  observation, keyed by the identity triple
  *series identity + target period + publication instant*;
* :class:`VintageAwareSplitter` - splitter wrapping any sktime temporal
  splitter, producing folds that contain only values visible at the fold's
  point-in-time cutoff;
* :func:`vintage_evaluate` - point-in-time correct backtest, reusing the
  exact fold evaluation and parallelization machinery of
  :func:`sktime.forecasting.model_evaluation.evaluate`;
* :func:`revision_report`, :func:`revision_summary`, :func:`revision_impact` -
  diagnostics showing which periods were revised, by how much, and how the
  revisions change backtest conclusions.

When only a final snapshot is available (no version information), wrap it with
:meth:`VintageFrame.from_snapshot`; the resulting backtest is identical, fold
by fold, to the ordinary ``evaluate`` backtest on the same snapshot.
"""

__author__ = ["sktime developers"]
__all__ = [
    "InsufficientVintageDataError",
    "VintageAwareSplitter",
    "VintageFrame",
    "VintageValidationError",
    "revision_impact",
    "revision_report",
    "revision_summary",
    "vintage_evaluate",
]

from sktime.vintage._diagnostics import (
    revision_impact,
    revision_report,
    revision_summary,
)
from sktime.vintage._evaluate import vintage_evaluate
from sktime.vintage._exceptions import (
    InsufficientVintageDataError,
    VintageValidationError,
)
from sktime.vintage._frame import VintageFrame
from sktime.vintage._splitter import VintageAwareSplitter
