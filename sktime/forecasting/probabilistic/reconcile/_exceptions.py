# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Exceptions for hierarchical probabilistic forecast reconciliation.

All exceptions carry structured, human-readable diagnostics so that callers
can pinpoint *which* node, level, time point or quantile is problematic,
instead of receiving a generic numerical failure.
"""

__all__ = [
    "HierarchyAdditivityError",
    "QuantileCrossingError",
    "ReconciliationError",
]


class ReconciliationError(ValueError):
    """Base class for errors raised during probabilistic reconciliation.

    Subclasses add structured ``details`` attributes describing the exact
    location of the problem (level, node, time point, quantile).
    """

    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        # always a list of dicts, one entry per offending location
        self.details = details if details is not None else []


class HierarchyAdditivityError(ReconciliationError):
    """Raised when a hierarchy is not additive under the given weights.

    A hierarchy is additive if, for every parent, the sum of the weights on
    the edges to its direct children equals one.

    ``details`` lists every offending parent with its level, child count and
    the actual child-weight sum.
    """


class QuantileCrossingError(ReconciliationError):
    """Raised when input quantiles/interval ends are not ordered.

    Quantiles must be isotonic in their probability level at every
    (node, time point): a lower quantile must not exceed a higher quantile.

    ``details`` lists every offending (node, time point, lower alpha,
    upper alpha) tuple with the two values.
    """
