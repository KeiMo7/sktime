"""Test-environment shim: restore sklearn.utils.extmath.stable_cumsum.

The editable sklearn in this environment is a 1.10 dev build where
``stable_cumsum`` was removed (deprecated since 1.8). The sktime checkout
under test still imports it. This shim re-adds the reference implementation
so the vintage module tests can run. It is NOT part of the sktime package.
"""

import numpy as np


def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    arr = np.asarray(arr)
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(
        np.isclose(
            out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
        )
    ):
        import warnings

        warnings.warn(
            "cumsum was found to be unstable: its last element does not "
            "correspond to sum",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


import sklearn.utils.extmath as _extmath

if not hasattr(_extmath, "stable_cumsum"):
    _extmath.stable_cumsum = stable_cumsum
