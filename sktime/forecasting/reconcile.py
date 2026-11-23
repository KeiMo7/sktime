# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements reconciled forecasters for hierarchical data."""

__all__ = ["ReconcilerForecaster"]
__author__ = ["ciaran-g", "felipeangelimvieira"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.warnings import warn


class ReconcilerForecaster(BaseForecaster):
    """Hierarchical reconciliation forecaster.

    Reconciliation is applied to make the forecasts in a hierarchy of
    time-series sum together appropriately.

    The base forecasts are first generated for each member separately in the
    hierarchy using any forecaster. The base forecasts are then reonciled
    so that they sum together appropriately. This reconciliation step can often
    improve the skill of the forecasts in the hierarchy.

    Please refer to [1]_ for further information.

    Parameters
    ----------
    forecaster : estimator
        Estimator to generate base forecasts which are then reconciled
    method : {"mint_cov", "mint_shrink", "ols", "wls_var", "wls_str", \
            "bu", "td_fcst"}, default="mint_shrink"
        The reconciliation approach applied to the forecasts based on:

            * ``"mint_cov"`` - sample covariance
            * ``"mint_shrink"`` - covariance with shrinkage
            * ``"ols"`` - ordinary least squares
            * ``"wls_var"`` - weighted least squares (variance)
            * ``"wls_str"`` - weighted least squares (structural)
            * ``"bu"`` - bottom-up
            * ``"td_fcst"`` - top down based on forecast proportions

    return_totals : bool
        Whether the predictions returned by ``predict`` and predict-like methods
        should include the total values in the hierarchy, stored at the ``__total``
        index levels.

        * If True, prediction data frames include total values at ``__total`` levels
        * If False, prediction data frames are returned without ``__total`` levels

    alpha: float default=0
        Optional regularization parameter to avoid singular covariance matrix

    See Also
    --------
    Aggregator
    Reconciler

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html

    Examples
    --------
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.reconcile import ReconcilerForecaster
    >>> from sktime.transformations.hierarchical.aggregate import Aggregator
    >>> from sktime.utils._testing.hierarchical import _bottom_hier_datagen
    >>> agg = Aggregator()
    >>> y = _bottom_hier_datagen(
    ...     no_bottom_nodes=3,
    ...     no_levels=1,
    ...     random_seed=123,
    ...     length=7,
    ... )
    >>> y = agg.fit_transform(y)
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> reconciler = ReconcilerForecaster(forecaster, method="mint_shrink")
    >>> reconciler.fit(y)
    ReconcilerForecaster(...)
    >>> prds_recon = reconciler.predict(fh=[1])
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ciaran-g", "felipeangelimvieira"],
        "maintainers": ["ciaran-g", "felipeangelimvieira"],
        # estimator type
        # --------------
        "category": ["composition"],
        "capability:multivariate": False,  # which y are fine? False/True
        "capability:exogenous": True,  # does estimator ignore the exogenous X?
        "capability:missing_values": False,  # can estimator handle missing data?
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd.Series",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd.Series",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,  # does forecaster implement proba forecasts?
        "capability:unequal_length": False,
        "capability:update": True,  # can estimator update its parameters with new data?
        "fit_is_empty": False,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def _decode_reconciler_method(self, method):
        """Decode the reconciliation method and return the corresponding class.

        Parameters
        ----------
        method : str
            The reconciliation method to decode.

        Returns
        -------
        Class : class
            The class corresponding to the reconciliation method.
        """
        from sktime.transformations.hierarchical.reconcile import (
            BottomUpReconciler,
            NonNegativeOptimalReconciler,
            OptimalReconciler,
            TopdownReconciler,
        )

        # We do not create the instances to avoid error due to
        # soft dependency on cvxpy for NonNegativeOptimalReconciler
        TRFORM_METHOD_MAP = {
            "bu": (BottomUpReconciler, {}),
            "ols": (OptimalReconciler, {}),
            "ols:nonneg": (NonNegativeOptimalReconciler, {}),
            "wls_str": (OptimalReconciler, {"error_covariance_matrix": "wls_str"}),
            "wls_str:nonneg": (
                NonNegativeOptimalReconciler,
                {"error_covariance_matrix": "wls_str"},
            ),
            "td_fcst": (TopdownReconciler, {"method": "td_fcst"}),
            "td_share": (TopdownReconciler, {"method": "td_share"}),
        }
        return TRFORM_METHOD_MAP[method]

    METHOD_LIST = [
        "mint_cov",
        "mint_shrink",
        "wls_var",
        "ols",
        "wls_str",
        "bu",
        "td_fcst",
        "td_share",
        "ols:nonneg",
        "wls_str:nonneg",
    ]
    RETURN_TOTALS_LIST = [True, False]

    def __init__(
        self,
        forecaster,
        method="mint_shrink",
        return_totals=True,
        alpha=0,
        reconcile_pred_int=False,
        pred_int_method=None,
        bounds=None,
        on_input_crossing="raise",
    ):
        self.forecaster = forecaster
        self.method = method
        self.return_totals = return_totals
        self.alpha = alpha
        # probabilistic reconciliation is opt-in; when False the estimator
        # behaves bit-for-bit like the previous point-only reconciler
        self.reconcile_pred_int = reconcile_pred_int
        self.pred_int_method = pred_int_method
        self.bounds = bounds
        self.on_input_crossing = on_input_crossing

        super().__init__()

        if self.reconcile_pred_int:
            # advertise probabilistic capability only when switched on,
            # so the default point-only behaviour remains unchanged
            self.set_tags(**{"capability:pred_int": True})

    def _check_method(self):
        """Raise warning if method is not defined correctly."""
        if not np.isin(self.method, self.METHOD_LIST):
            raise ValueError(f"""method must be one of {self.METHOD_LIST}.""")
        else:
            pass

    def _get_prob_reconciler_method(self):
        """Map the public method choice to a probabilistic strategy name."""
        if self.pred_int_method is not None:
            return self.pred_int_method
        # reuse the point-forecast strategy where an interval analogue exists
        mapping = {
            "bu": "bu",
            "td_fcst": "td_fcst",
            "td_share": "td_share",
            "ols": "ols",
            "ols:nonneg": "ols",
            "wls_str": "wls_str",
            "wls_str:nonneg": "wls_str",
            "wls_var": "wls_var",
            "mint_cov": "mint_cov",
            "mint_shrink": "mint_shrink",
        }
        return mapping[self.method]

    def _add_totals(self, y):
        """Add total levels to y, using Aggregate."""
        from sktime.transformations.hierarchical.aggregate import Aggregator

        return Aggregator().fit_transform(y)

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Time series to which to fit the forecaster.
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, default=None
            Exogenous variables for the base forecaster

        Returns
        -------
        self : reference to self
        """
        self._check_method()

        # # check the length of index if not hierarchical just return self early
        if y.index.nlevels < 2:
            self.forecaster_ = self.forecaster.clone()
            self.forecaster_.fit(y=y, X=X, fh=fh)
            self._prob_reconciler_ = None
            return self

        self._series_to_keep = y.index.droplevel(-1).unique()

        from sktime.transformations.hierarchical.aggregate import _check_index_no_total

        # Add totals and flatten single levels
        if _check_index_no_total(y):
            y = self._add_totals(y)

        if X is not None:
            if _check_index_no_total(X):
                X = self._add_totals(X)

        if self.return_totals:
            # Override the series to keep
            self._series_to_keep = y.index.droplevel(-1).unique()

        self.forecaster_ = self.forecaster.clone()

        if not self._requires_residuals:
            Class, kwargs = self._decode_reconciler_method(self.method)
            self.reconciler_transform_ = Class(**kwargs)
            yt = self.reconciler_transform_.fit_transform(y)
            self.forecaster_.fit(y=yt, X=X, fh=fh)
            self._fit_probabilistic_reconciler(y, residuals=None)
            return self
        # fit forecasters for each level

        # In this case, the totals are required
        y = self._add_totals(y)
        self.forecaster_.fit(y=y, X=X, fh=fh)
        # bug in self.forecaster_.predict_residuals() for heir data
        fh_resid = ForecastingHorizon(
            y.index.get_level_values(-1).unique(), is_relative=False
        )
        self.residuals_ = y - self.forecaster_.predict(fh=fh_resid, X=X)

        # now define recon matrix
        if self.method == "mint_cov":
            self.error_cov_matrix_ = self._get_error_covariance_matrix(shrink=False)
        elif self.method == "mint_shrink":
            self.error_cov_matrix_ = self._get_error_covariance_matrix(shrink=True)
        elif self.method == "wls_var":
            self.error_cov_matrix_ = self._get_error_covariance_matrix(
                shrink=False, diag_only=True
            )

        from sktime.transformations.hierarchical.reconcile import (
            NonNegativeOptimalReconciler,
            OptimalReconciler,
        )

        ReconcilerClass = OptimalReconciler
        if self.method.endswith(":nonneg"):
            ReconcilerClass = NonNegativeOptimalReconciler
        self.reconciler_transform_ = ReconcilerClass(
            self.error_cov_matrix_, alpha=self.alpha
        )
        self.reconciler_transform_.fit(y)

        self._fit_probabilistic_reconciler(y)
        return self

    def _fit_probabilistic_reconciler(self, y, residuals=None):
        """Build the probabilistic reconciler if the feature is enabled.

        No-op when ``reconcile_pred_int`` is False, so the default point
        reconciliation path keeps its exact behaviour.
        """
        if not self.reconcile_pred_int:
            self._prob_reconciler_ = None
            return
        from sktime.forecasting.probabilistic.reconcile import (
            ProbabilisticHierarchicalReconciler,
        )

        prob_method = self._get_prob_reconciler_method()
        # non-negativity point strategies carry over into interval bounds
        bounds = self.bounds
        if bounds is None and self.method.endswith(":nonneg"):
            bounds = (0.0, None)
        self._prob_reconciler_ = ProbabilisticHierarchicalReconciler(
            method=prob_method,
            bounds=bounds,
            on_input_crossing=self.on_input_crossing,
        )
        resid = residuals
        if resid is None and hasattr(self, "residuals_"):
            resid = self.residuals_
        self._prob_reconciler_.fit(y_train=y, residuals=resid)
        # remember the frame layout needed to align predict outputs
        self._prob_node_index_ = y.index.droplevel(-1).unique()

    def _expand_base_quantiles(self, base_q):
        """Structurally expand base quantiles to the full hierarchy.

        Pipeline reconcilers (e.g. bottom-up) fit the inner forecaster on
        bottom-level series only, so its quantile output lacks aggregate
        nodes. This re-adds them by plain summation through the
        ``Aggregator``, without applying any point-reconciliation matrix,
        leaving the probabilistic reconciler in charge of the strategy.
        Quantile sums are coherent because summation is linear.
        """
        from sktime.transformations.hierarchical.aggregate import Aggregator

        frame_nodes = base_q.index.droplevel(-1).unique()
        full_nodes = self._prob_node_index_
        if len(frame_nodes) >= len(full_nodes):
            return base_q.sort_index()

        var_names = base_q.columns.get_level_values(0).unique()
        expanded_blocks = []
        for var in var_names:
            alphas = base_q[var].columns
            blocks = []
            for a in alphas:
                col = base_q.loc[:, (var, a)].to_frame()
                col.columns = [var]
                col = Aggregator(
                    flatten_single_levels=False
                ).fit_transform(col)
                block = col.rename(columns={var: a})
                block.columns = pd.MultiIndex.from_product(
                    [[var], [a]]
                )
                blocks.append(block.sort_index())
            expanded_blocks.append(pd.concat(blocks, axis=1))
        return pd.concat(expanded_blocks, axis=1).sort_index()

    def _predict_quantiles(self, fh, X, alpha):
        """Compute reconciled hierarchical quantile forecasts.

        Only reachable when ``reconcile_pred_int=True`` (the
        ``capability:pred_int`` tag is switched off otherwise).
        """
        if getattr(self, "_prob_reconciler_", None) is None:
            # single-series path: delegate unchanged
            return self.forecaster_.predict_quantiles(fh=fh, X=X, alpha=alpha)

        from sktime.transformations.hierarchical.aggregate import _check_index_no_total

        if X is not None and _check_index_no_total(X):
            X = self._add_totals(X)

        base_q = self.forecaster_.predict_quantiles(fh=fh, X=X, alpha=alpha)

        if base_q.index.nlevels < 2:
            return base_q

        # expand bottom-only base forecasts to the full hierarchy first
        base_q = self._expand_base_quantiles(base_q)

        rec_frames = []
        info = None
        for var in base_q.columns.get_level_values(0).unique():
            var_frame = base_q[[var]]
            rec_frame, info = (
                self._prob_reconciler_.reconcile_quantiles_frame(
                    var_frame, var_name=var
                )
            )
            rec_frames.append(rec_frame)
        reconc = pd.concat(rec_frames, axis=1)

        # align row layout with the point predict output contract
        if not self.return_totals:
            from sktime.transformations.hierarchical.aggregate import Aggregator

            agg = Aggregator(
                False, bypass_inverse_transform=False
            ).fit(reconc)
            reconc = agg.inverse_transform(reconc)
        # diagnostics are exposed through the reconciler component, not as
        # a mutable attribute on this estimator (keeps __dict__ stable)
        return reconc.sort_index()

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : optional (default=None)
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        from sktime.transformations.hierarchical.aggregate import _check_index_no_total

        if X is not None:
            if _check_index_no_total(X):
                X = self._add_totals(X)

        base_fc = self.forecaster_.predict(fh=fh, X=X)

        if base_fc.index.nlevels < 2:
            warn(
                "Reconciler is intended for use with y.index.nlevels > 1. "
                "Returning predictions unchanged.",
                obj=self,
            )
            return base_fc

        reconc_fc = self.reconciler_transform_.inverse_transform(base_fc)

        from sktime.transformations.hierarchical.aggregate import Aggregator
        from sktime.transformations.hierarchical.reconcile._utils import (
            _loc_series_idxs,
        )

        agg = Aggregator(False, bypass_inverse_transform=False).fit(reconc_fc)
        if not self.return_totals:
            return agg.inverse_transform(reconc_fc)

        reconc_fc = agg.fit_transform(reconc_fc)
        reconc_fc = _loc_series_idxs(reconc_fc, self._series_to_keep)

        return reconc_fc

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        Parameters
        ----------
        y : pd.DataFrame
            Time series to which to fit the forecaster.
        X : pd.DataFrame, default=None
            Exogenous variables based to the base forecaster
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        from sktime.transformations.hierarchical.aggregate import _check_index_no_total

        # check index for no "__total", if not add totals to y
        if _check_index_no_total(y):
            y = self._add_totals(y)

        if X is not None:
            if _check_index_no_total(X):
                X = self._add_totals(X)

        if y.index.nlevels < 2 or not self._requires_residuals:
            self.forecaster_.update(y, X, update_params=update_params)
            return self

        # update self.residuals_
        # bug in self.forecaster_.predict_residuals() for heir data
        fh_resid = ForecastingHorizon(
            y.index.get_level_values(-1).unique(), is_relative=False
        )
        update_residuals = y - self.forecaster_.predict(fh=fh_resid, X=X)
        self.forecaster_.update(y, X, update_params=update_params)
        self.residuals_ = pd.concat([self.residuals_, update_residuals], axis=0)
        self.residuals_ = self.residuals_.sort_index()

        # now define recon matrix
        if self.method == "mint_cov":
            self.error_cov_matrix_ = self._get_error_covariance_matrix(shrink=False)
        elif self.method == "mint_shrink":
            self.error_cov_matrix_ = self._get_error_covariance_matrix(shrink=True)
        elif self.method == "wls_var":
            self.error_cov_matrix_ = self._get_error_covariance_matrix(
                shrink=False, diag_only=True
            )

        return self

    @property
    def _requires_residuals(self):
        """Check if the reconciliation requires residuals."""
        return self.method in ["mint_cov", "mint_shrink", "wls_var"]

    def _get_error_covariance_matrix(self, shrink=False, diag_only=False):
        """Get the error covariance matrix for the MinT methods.

        Reconciliation methods require the error covariance matrix.
        The error covariance matrix is used to define the covariance of the
        residuals for the entire hierarchy
        to the bottom-level only before summation using the S matrix.

        Please refer to [1]_ for further information.

        Parameters
        ----------
        shrink:  bool, optional (default=False)
            Shrink the off diagonal elements of the sample covariance matrix.
            according to the method in [2]_
        diag_only: bool, optional (default=False)
            Remove the off-diagonal elements of the sample covariance matrix.

        Returns
        -------
        cov_mint : pd.DataFrame with rows equal to the number of bottom level nodes
            only, i.e. with no aggregate nodes, and columns equal to the number of
            unique nodes in the hierarchy. The matrix indexes is inherited from the
            input data, with the time level removed.

        References
        ----------
        .. [1] https://otexts.com/fpp3/hierarchical.html
        .. [2] https://doi.org/10.2202/1544-6115.1175
        """
        if self.residuals_.index.nlevels < 2:
            return None

        # copy for further mods
        resid = self.residuals_.copy()
        resid = resid.unstack().transpose()
        cov_mat = resid.cov()

        if self.residuals_.index.nlevels < 2:
            return None

        # copy for further mods
        resid = self.residuals_.copy()
        resid = resid.unstack().transpose()
        cov_mat = resid.cov()

        if shrink:
            # diag matrix of variances
            diag_mat = np.diag(np.diag(cov_mat.values))
            var_d = pd.DataFrame(diag_mat, index=cov_mat.index, columns=cov_mat.columns)

            # get correlation from covariance above
            cor_mat = resid.corr()
            nobs = len(resid)

            # first standardize the residuals
            resid = resid.apply(lambda x: (x - x.mean()) / x.std())

            # scale for higher order var calc
            scale_hovar = ((resid.transpose().dot(resid)) ** 2) * (1 / nobs)

            # higherorder var (only diags)
            resid_corseries = resid**2
            hovar_mat = (resid_corseries.transpose().dot(resid_corseries)) - scale_hovar
            hovar_mat = (nobs / (nobs - 1) ** 3) * hovar_mat

            # set diagonals to zero
            for i in resid.columns:
                hovar_mat.loc[hovar_mat.index == i, hovar_mat.columns == i] = 0
                cor_mat.loc[cor_mat.index == i, cor_mat.columns == i] = 0

            # get the shrinkage value
            lamb = hovar_mat.sum().sum() / (cor_mat**2).sum().sum()
            lamb = np.min([1, np.max([0, lamb])])

            # shrink the matrix
            cov_mat = (lamb * var_d) + ((1 - lamb) * cov_mat)

        if diag_only:
            # digonal matrix of variances
            for i in resid.columns:
                cov_mat.loc[cov_mat.index != i, cov_mat.columns == i] = 0

        return cov_mat

    def _check_method(self):
        """Raise warning if method is not defined correctly."""
        if not np.isin(self.method, self.METHOD_LIST):
            raise ValueError(f"""method must be one of {self.METHOD_LIST}.""")
        else:
            pass

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for clusterers.

        Returns
        -------
        params : dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.forecasting.trend import TrendForecaster

        FORECASTER = TrendForecaster()
        methods_without_soft_deps = [
            x for x in cls.METHOD_LIST if not x.endswith("nonneg")
        ]
        params_list = [
            {
                "forecaster": FORECASTER,
                "method": x,
                "alpha": alpha,
                "return_totals": totals,
            }
            for x in methods_without_soft_deps
            for totals in cls.RETURN_TOTALS_LIST
            for alpha in [0, 1]
        ]
        return params_list
