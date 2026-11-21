#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Exceptions for the vintage (point-in-time data) module."""

__author__ = ["sktime developers"]
__all__ = ["VintageValidationError", "InsufficientVintageDataError"]


class VintageValidationError(ValueError):
    """Raised when a vintage frame fails structural validation.

    Covers null identity components, duplicate identity triples, publication
    instants predating the target period, broken version chains and null
    observed values. Error messages identify the offending series and period.
    """


class InsufficientVintageDataError(ValueError):
    """Raised when a backtest origin has too little visible data.

    This is raised (or, depending on the ``on_insufficient`` setting, avoided
    in favour of skipping the fold) when no training observation or no test
    observation is visible at a fold's point-in-time cutoff. Future data is
    never used to fill the gap.
    """
