# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Core engine for hierarchical probabilistic forecast reconciliation.

Given base (unreconciled) point forecasts, prediction intervals or quantiles
for every node of a hierarchy, produce reconciled forecasts whose interval
ends are exactly coherent across levels, while preserving quantile order
(non-crossing) and optional box constraints (non-negativity, capacities).

Design
------
Two numerical paths are used, per time point and per quantile level:

1. **Linear path (default, fully vectorised):** reconciled values are
   ``P x`` with a projection matrix ``P = S G`` derived from the chosen
   strategy (bottom-up, top-down, OLS, WLS, MinT). For the structural
   strategies (bottom-up, top-down, WLS structural) the projection matrix is
   non-negative, so quantile order is preserved automatically; coherence
   holds exactly by construction.

2. **Constrained quadratic path (only when needed):** when the linear path
   cannot satisfy all requirements -- input crossings, explicit bounds or
   custom weights -- a convex quadratic program is solved with scipy
   (no network service). Reconciliation is parameterised through the bottom
   nodes only (``y = S_w beta``), so the coherence equalities vanish from
   the solver and are exact regardless of the solver tolerance. Monotone
   differences between adjacent quantiles are constrained non-negative,
   which rules out zero-width intervals unless the bounds force one.

The engine is deterministic: identical inputs and parameters yield
bit-identical outputs (no random initialisation, no parallel scheduling).
"""

__all__ = ["ProbabilisticHierarchicalReconciler"]

import numpy as np
import pandas as pd

from sktime.forecasting.probabilistic.reconcile._exceptions import (
    ReconciliationError,
)
from sktime.forecasting.probabilistic.reconcile._structure import (
    HierarchyStructure,
)
from sktime.forecasting.probabilistic.reconcile._validation import (
    array_to_quantile_frame,
    find_quantile_crossings,
    quantile_frame_to_array,
    raise_on_crossings,
)

VALID_METHODS = (
    "bu",
    "td_fcst",
    "td_share",
    "ols",
    "wls_str",
    "wls_var",
    "mint_cov",
    "mint_shrink",
)

# minimum residual history (relative to number of nodes) for a full
# covariance estimate; below this we degrade the strategy transparently
MIN_RESID_RATIO = 2.0


class ProbabilisticHierarchicalReconciler:
    """Reconcile point forecasts, intervals and quantiles across a hierarchy.

    Parameters
    ----------
    method : str, default="bu"
        Reconciliation strategy, one of:

        * ``"bu"`` - bottom-up: bottom nodes kept, aggregates rebuilt by sum
        * ``"td_fcst"`` - top-down via forecast proportions (per time point)
        * ``"td_share"`` - top-down via historical/forecast shares
        * ``"ols"`` - optimal reconciliation, identity error covariance
        * ``"wls_str"`` - weighted least squares with structural variances
        * ``"wls_var"`` - diagonal variance weighting (needs residuals)
        * ``"mint_cov"`` - MinT with full sample covariance (needs residuals)
        * ``"mint_shrink"`` - MinT with shrinkage covariance (needs residuals)

        When a covariance-based method lacks enough residual samples, the
        engine degrades to ``"ols"`` and records the fallback in
        ``method_used_`` / ``warnings_``.
    bounds : tuple of (lower, upper), optional
        Box constraints applied to every reconciled value, e.g.
        ``(0.0, None)`` for non-negativity, ``(0.0, 1000.0)`` for a capacity.
        Bounds are enforced through a convex QP that stays coherent; values
        are never clipped post hoc (clipping would break coherence).
    on_input_crossing : {"raise", "isotonic"}, default="raise"
        Behaviour when the *input* quantiles already cross:

        * ``"raise"`` (default): raise ``QuantileCrossingError`` with the
          full list of offending locations; nothing is modified.
        * ``"isotonic"``: repair jointly inside the constrained QP, keeping
          the result as close as possible to the input in weighted distance.
    solver : {"auto", "slsqp"}, default="auto"
        QP solver. Currently scipy's SLSQP is used (offline only).
    solver_tol : float, default=1e-8
        Termination tolerance for the constrained path.
    """

    def __init__(
        self,
        method="bu",
        bounds=None,
        on_input_crossing="raise",
        solver="auto",
        solver_tol=1e-8,
    ):
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {VALID_METHODS}, got {method!r}"
            )
        if on_input_crossing not in ("raise", "isotonic"):
            raise ValueError(
                "on_input_crossing must be 'raise' or 'isotonic', got "
                f"{on_input_crossing!r}"
            )
        if bounds is not None:
            if not (isinstance(bounds, tuple) and len(bounds) == 2):
                raise ValueError(
                    "bounds must be a (lower, upper) tuple, e.g. (0.0, None)"
                )
        self.method = method
        self.bounds = bounds
        self.on_input_crossing = on_input_crossing
        self.solver = solver
        self.solver_tol = solver_tol

        # populated by fit
        self.warnings_ = []
        self.method_used_ = method
        self._projection = None  # set when a fixed linear projection exists

    # ================================================================== #
    # fit: learn structure, linear projection, covariance weighting
    # ================================================================== #
    def fit(self, y_train=None, node_index=None, weights=None, residuals=None):
        """Learn the hierarchy structure and reconciliation projection.

        Parameters
        ----------
        y_train : pd.DataFrame, optional
            Hierarchical training frame (sktime multi-index with a
            ``"__total"`` aggregate convention). Used to infer node order.
        node_index : pd.MultiIndex, optional
            Explicit hierarchy node index (alternative to ``y_train``).
        weights : pd.DataFrame, optional
            Edge weights (see ``HierarchyStructure``).
        residuals : pd.DataFrame, optional
            In-sample residuals with the same hierarchical index shape as
            ``y_train``; required for covariance-based strategies.

        Returns
        -------
        self
        """
        if node_index is None:
            if y_train is None:
                raise ValueError(
                    "fit requires either y_train or node_index to infer the "
                    "hierarchy structure."
                )
            node_index = y_train.index.droplevel(-1).unique()

        self.structure_ = HierarchyStructure(node_index, weights=weights)
        self.node_index_ = self.structure_.S_.index
        self._S = self.structure_.aggregation_matrix().values.astype(float)
        self._S_bool = self.structure_.S_.values.astype(float)

        # covariance / weighting matrix over all nodes
        self._E = self._estimate_error_covariance(residuals)

        # QP-only building blocks: build lazily so purely linear
        # reconciliation (bu / crossing-free ols, wls) pays no SVD,
        # eigendecomposition or extra pinv at fit time
        self._qp_W = None
        self._qp_StW = None
        self._qp_SW = None
        self._qp_lipschitz = None
        self._S_pinv = None
        self._col_space_projector = None

        # fixed linear projections (time-independent strategies)
        if self.method in ("bu", "ols", "wls_str"):
            self._projection = self._linear_projection(self.method_used_)
        elif self.method in ("wls_var", "mint_cov", "mint_shrink"):
            self._projection = self._linear_projection(self.method_used_)
        # td_* projections are forecast-dependent, rebuilt per call
        return self

    def _ensure_qp_matrices(self):
        """Materialise QP weighting blocks on first constrained solve."""
        if self._qp_SW is not None:
            return
        W = np.linalg.pinv(self._E)
        self._qp_W = 0.5 * (W + W.T)
        self._qp_StW = self._S.T @ self._qp_W
        self._qp_SW = 0.5 * (
            self._qp_StW @ self._S
            + (self._qp_StW @ self._S).T
        )
        self._qp_lipschitz = float(
            max(np.linalg.eigvalsh(2.0 * self._qp_SW).max(), 1e-12)
        )

    @property
    def _pinv_S(self):
        if self._S_pinv is None:
            self._S_pinv = np.linalg.pinv(self._S)
        return self._S_pinv

    def _get_col_space_projector(self):
        if self._col_space_projector is None:
            self._col_space_projector = self._S @ self._pinv_S
        return self._col_space_projector

    def _estimate_error_covariance(self, residuals):
        """Build the node-level error covariance matrix used for weighting."""
        n = self.structure_.n_nodes
        if residuals is None:
            if self.method in ("wls_var", "mint_cov", "mint_shrink"):
                self._degrade(
                    "no residuals supplied; covariance-based strategy "
                    f"{self.method!r} degraded to 'ols'."
                )
            return np.eye(n)

        resid = residuals.copy()
        # wide frame: rows = time, columns = nodes
        wide = resid.unstack().droplevel(0, axis=1)
        wide = wide.reindex(columns=self.node_index_)
        n_resid = wide.shape[0]

        if self.method in ("mint_cov", "mint_shrink"):
            if n_resid < max(MIN_RESID_RATIO * n, 3):
                self._degrade(
                    f"only {n_resid} residual time points for {n} nodes "
                    f"(need >= {max(int(MIN_RESID_RATIO * n), 3)}); "
                    f"{self.method!r} degraded to 'ols'."
                )
                return np.eye(n)
            cov = np.cov(wide.values, rowvar=False)
            if self.method == "mint_shrink":
                cov = _shrink_covariance(wide.values, cov)
            cov = cov + 1e-12 * np.eye(n)
            return cov

        if self.method == "wls_var":
            if n_resid < 2:
                self._degrade(
                    "fewer than 2 residual time points; 'wls_var' degraded "
                    "to 'ols'."
                )
                return np.eye(n)
            var = wide.var(axis=0, ddof=1).values
            var = np.where(var > 1e-12, var, 1.0)
            return np.diag(var)

        return np.eye(n)

    def _degrade(self, message):
        self.warnings_.append(message)
        self.method_used_ = "ols"

    def _linear_projection(self, method):
        """Compute P = S_w G for time-independent linear strategies.

        All inversions are carried out in the *bottom* space of dimension
        ``n_bottom`` (typically much smaller than the full node count),
        which keeps fit tractable for panels with thousands of leaves.
        """
        S = self._S
        n_nodes, n_bottom = S.shape
        if method == "bu":
            # keep bottom rows, rebuild aggregates
            G = np.zeros((n_bottom, n_nodes))
            bottom_cols = list(self.structure_.bottom_nodes)
            for j, node in enumerate(bottom_cols):
                G[j, list(self.node_index_).index(node)] = 1.0
            return S @ G

        if method == "ols":
            # G = (S' S)^-1 S' (bottom-space inverse only)
            G = np.linalg.solve(S.T @ S, S.T)
            return S @ G

        if method == "wls_str":
            # structural variances: node variance proportional to its
            # number of contributing bottom nodes
            diag = self._S_bool.sum(axis=1)
            W_inv = np.diag(1.0 / np.maximum(diag, 1e-12))
            G = np.linalg.solve(S.T @ W_inv @ S, S.T @ W_inv)
            return S @ G

        if method in ("wls_var", "mint_cov", "mint_shrink"):
            E_inv = np.linalg.pinv(self._E)
            E_inv = 0.5 * (E_inv + E_inv.T)
            # normalise to avoid scale-induced ill-conditioning
            scale = float(np.mean(np.diag(E_inv)))
            if scale > 0:
                E_inv = E_inv / scale
            G = np.linalg.solve(S.T @ E_inv @ S, S.T @ E_inv)
            return S @ G

        raise ReconciliationError(f"no linear projection for method {method!r}")

    # ================================================================== #
    # public reconciliation entry points
    # ================================================================== #
    def reconcile_quantiles_frame(self, quantile_frame, var_name=None):
        """Reconcile an sktime-style hierarchical quantile frame.

        Parameters
        ----------
        quantile_frame : pd.DataFrame
            Output of ``BaseForecaster.predict_quantiles``.
        var_name : str, optional
            Variable to reconcile; defaults to the first variable.

        Returns
        -------
        reconciled : pd.DataFrame
            Same index/columns layout as the input; every column level is
            coherent across hierarchy levels and quantiles are non-crossing.
        info : dict
            Diagnostics: method used, fallback warnings, width statistics,
            and which rows required the constrained solver.
        """
        if not hasattr(self, "structure_"):
            raise RuntimeError("call fit before reconcile_quantiles_frame")

        var_names = list(
            quantile_frame.columns.get_level_values(0).unique()
        )
        if var_name is None:
            var_name = var_names[0]
        alphas = list(
            quantile_frame.columns.get_level_values(1).unique()
        )
        alphas = sorted(float(a) for a in alphas)

        arr, time_index = quantile_frame_to_array(
            quantile_frame, self.node_index_, alphas, var_name
        )

        crossings = find_quantile_crossings(arr, alphas)
        if crossings and self.on_input_crossing == "raise":
            raise_on_crossings(
                crossings, self.node_index_, time_index, var_name
            )

        rec_arr, info = self._reconcile_array(
            arr, time_index, repair_crossing=bool(crossings)
        )

        out = array_to_quantile_frame(
            rec_arr, quantile_frame, self.node_index_, time_index, alphas
        )
        info["alphas"] = alphas
        info["variable"] = var_name
        info["warnings"] = list(self.warnings_)
        info["method_used"] = self.method_used_
        return out, info

    def reconcile_points_frame(self, point_frame):
        """Reconcile a hierarchical point forecast (single "quantile")."""
        if not hasattr(self, "structure_"):
            raise RuntimeError("call fit before reconcile_points_frame")
        frame = point_frame.sort_index()
        time_index = frame.index.get_level_values(-1).unique()
        wide = (
            frame.iloc[:, 0]
            .unstack()
            .reindex(columns=self.node_index_)
            .reindex(index=time_index)
        )
        x = wide.values[:, :, None]  # (T, N, 1)
        rec, _ = self._reconcile_array(x, time_index, repair_crossing=False)
        rec_wide = pd.DataFrame(
            rec[:, :, 0], index=time_index, columns=self.node_index_
        )
        long_df = rec_wide.stack().swaplevel().sort_index().to_frame(frame.columns[0])
        long_df.index.names = frame.index.names
        return long_df

    # ================================================================== #
    # core array machinery
    # ================================================================== #
    def _reconcile_array(self, x, time_index, repair_crossing):
        """Reconcile numeric array of shape (n_time, n_nodes, n_quantiles).

        The linear projection is applied first (fast, vectorised). Its
        projection matrix can carry negative weights for OLS/WLS/MinT, so
        crossing and box constraints are re-checked afterwards; any time
        point that fails is re-solved with the constrained QP. The output
        is therefore always coherent, non-crossing and within bounds.
        """
        n_t, n_n, n_q = x.shape
        need_bounds = self.bounds is not None
        width_before = _mean_width(x)
        qp_rows = []

        # ---- fast linear path -------------------------------------------
        if self._projection is not None:
            linear = np.einsum("ij,tjk->tik", self._projection, x)
        elif self.method in ("td_fcst", "td_share"):
            linear = self._topdown_linear(x, time_index)
        else:  # pragma: no cover
            linear = np.einsum("ij,tjk->tik", self._fallback_projection(), x)

        # decide which time points the linear path cannot satisfy
        rec = linear.copy()
        if n_q > 1:
            crossed_t = np.where(
                np.diff(linear, axis=-1).min(axis=(1, 2)) < -1e-8
            )[0]
        else:
            crossed_t = np.array([], dtype=int)
        if need_bounds:
            lb, ub = self.bounds
            low_t = (
                np.where(linear.min(axis=(1, 2)) < lb - 1e-8)[0]
                if lb is not None
                else np.array([], dtype=int)
            )
            high_t = (
                np.where(linear.max(axis=(1, 2)) > ub + 1e-8)[0]
                if ub is not None
                else np.array([], dtype=int)
            )
            bound_t = np.unique(np.concatenate([low_t, high_t]))
        else:
            bound_t = np.array([], dtype=int)
        problem_t = np.unique(np.concatenate([crossed_t, bound_t]))
        # explicit input-crossing repair always uses the joint QP
        if repair_crossing:
            problem_t = np.arange(n_t)

        for t in problem_t:
            rec[t], _ = self._solve_timepoint(
                x[t], self._fallback_projection(), t, time_index, qp_rows
            )

        # final safety checks (must always hold)
        if len(problem_t) > 0:
            # only the QP path needs the column-space projection check;
            # linear rows are P x = S G x, hence exactly in col(S) by design
            self._assert_coherent(rec, time_index, rows=problem_t)
        self._assert_no_crossing(rec, alphas=None)
        if need_bounds:
            self._assert_bounds(rec)

        width_after = _mean_width(rec)
        info = {
            "n_timepoints": n_t,
            "n_nodes": n_n,
            "n_quantiles": n_q,
            "mean_width_before": width_before,
            "mean_width_after": width_after,
            "width_ratio": (
                width_after / width_before if width_before > 0 else np.nan
            ),
            "constrained_solver_rows": qp_rows,
            "n_linear_rows": int(n_t - len(problem_t)),
            "bounds_applied": need_bounds,
            "input_crossing_repaired": repair_crossing,
        }
        return rec, info

    def _fallback_projection(self):
        """Projection used to warm-start / weight the constrained path."""
        if self._projection is not None:
            return self._projection
        return self._linear_projection(
            self.method_used_ if self.method_used_ in ("bu", "ols", "wls_str")
            else "ols"
        )

    @staticmethod
    def _monotonicity_matrix(n_bottom, n_q):
        """Cached sparse-ish matrix enforcing quantile monotonicity."""
        rows, cols, vals = [], [], []
        r = 0
        for j in range(n_bottom):
            for k in range(n_q - 1):
                rows += [r, r]
                cols += [j * n_q + k, j * n_q + k + 1]
                vals += [-1.0, 1.0]
                r += 1
        mono = np.zeros((n_bottom * (n_q - 1), n_bottom * n_q))
        mono[rows, cols] = vals
        return mono

    def _topdown_linear(self, x, time_index):
        """Time-point-wise top-down reconciliation preserving quantile order.

        Uses forecast proportions computed separately per quantile level;
        proportions are non-negative so ordering is preserved.
        """
        S = self._S
        n_t, n_n, n_q = x.shape
        rec = np.empty_like(x)
        parent_child = self.structure_.parent_child_
        node_list = list(self.node_index_)
        pos = {node: i for i, node in enumerate(node_list)}

        for t in range(n_t):
            for k in range(n_q):
                vals = x[t, :, k]
                # start from the top node and walk down
                beta = np.zeros(S.shape[1])
                # bottom-up total of the top-level base forecast is
                # distributed by forecast proportions at each parent
                bottom_nodes = list(self.structure_.bottom_nodes)
                # compute proportions recursively from base forecasts
                props = self._forecast_proportions(vals, pos, parent_child)
                top_idx = pos[_top_node(self.structure_)]
                total = vals[top_idx]
                for j, b in enumerate(bottom_nodes):
                    beta[j] = total * props.get(b, 0.0)
                rec[t, :, k] = S @ beta
        return rec

    def _forecast_proportions(self, vals, pos, parent_child):
        """Product of child/base-sum ratios along each root->leaf path."""
        parent_map = dict(
            zip(parent_child["child"], parent_child["parent"])
        )
        sibling_map = {
            parent: list(grp["child"])
            for parent, grp in parent_child.groupby("parent")
        }
        props = {}
        for bottom in self.structure_.bottom_nodes:
            child = bottom
            p = 1.0
            while child in parent_map:
                parent = parent_map[child]
                siblings = sibling_map[parent]
                denom = sum(vals[pos[s]] for s in siblings)
                if not np.isfinite(denom) or abs(denom) < 1e-12:
                    # equal split fallback, keeps proportions non-negative
                    p *= 1.0 / len(siblings)
                else:
                    p *= vals[pos[child]] / denom
                child = parent
            props[bottom] = p
        # normalise to guard against rounding drift
        total_p = sum(props.values())
        if abs(total_p - 1.0) > 1e-10 and total_p > 0:
            props = {k: v / total_p for k, v in props.items()}
        return props

    # ------------------------------------------------------------------ #
    # constrained QP
    # ------------------------------------------------------------------ #
    def _solve_timepoint(self, x_t, P0, t, time_index, qp_rows):
        """Solve the joint QP for one time point, all quantile levels.

        Decision variable: ``beta`` of shape ``(n_bottom, n_quantiles)``.
        Reconciled nodes are ``y = S beta`` so coherence is automatic.

        Objective (weighted distance to the linearly reconciled target
        ``z = P0 x``):
            sum_k (S beta_k - z_k)' W (S beta_k - z_k)
        Constraints:
            beta_{j,k+1} - beta_{j,k} >= 0   (quantile monotonicity)
            bounds on every node value, mapped onto bottom contributions

        For the standard non-negative summation matrix, all constraints are
        simple (box + per-row isotonicity), so a FISTA projected-gradient
        solver is used: every projection is closed form (PAV + clipping),
        there is no generic line search, and the result is exactly feasible.
        A scipy SLSQP path is retained for signed-weight hierarchies.
        """
        self._ensure_qp_matrices()
        S = self._S
        n_bottom, n_q = S.shape[1], x_t.shape[1]
        z = P0 @ x_t  # (n_nodes, n_q), coherent warm-start target

        W = self._qp_W
        SW = self._qp_SW  # S' W S, precomputed at fit time
        StW = self._qp_StW  # S' W

        lb, ub = self.bounds if self.bounds is not None else (None, None)
        s_nonneg = (S >= -1e-12).all()

        # initial guess: coherent + monotone + in-bounds bottom forecast
        beta0 = self._pinv_S @ z
        beta0 = _isotonic_rows(beta0)
        if s_nonneg:
            if lb is not None:
                beta0 = np.maximum(beta0, lb)
            if ub is not None:
                beta0 = np.minimum(beta0, ub)
            beta0 = _isotonic_rows(beta0)

            beta, converged = self._solve_fista(
                beta0, z, S, SW, StW, W, lb, ub
            )
            if not converged:
                # deterministic fallback within the same feasible set
                beta = self._solve_slsqp(
                    beta0, z, S, SW, StW, W, n_bottom, n_q, lb, ub
                )
        else:
            beta = self._solve_slsqp(
                beta0, z, S, SW, StW, W, n_bottom, n_q, lb, ub,
                explicit_node_bounds=True,
            )

        # exact feasibility re-assertion (projection is idempotent)
        beta = _isotonic_rows(beta)
        if s_nonneg:
            if lb is not None:
                beta = np.maximum(beta, lb)
            if ub is not None:
                beta = np.minimum(beta, ub)
        y = S @ beta
        qp_rows.append(str(time_index[t]))
        return y, True

    def _solve_fista(self, beta0, z, S, SW, StW, W, lb, ub,
                     max_iter=800, tol=1e-6):
        """FISTA projected gradient for box + isotonic constraints.

        Convergence is declared on the relative change of the objective,
        which is scale-free and reaches business precision well before the
        tighter step-size criterion; a minimum iteration guard avoids
        premature stops.
        """
        # Lipschitz constant precomputed at fit time
        L = self._qp_lipschitz

        def project(b):
            b = _isotonic_rows(b)
            if lb is not None:
                b = np.maximum(b, lb)
            if ub is not None:
                b = np.minimum(b, ub)
            return _isotonic_rows(b)

        def objective(b):
            diff = S @ b - z
            return float(np.einsum("nk,nm,mk->", diff, W, diff))

        beta = beta0.copy()
        accel = beta0.copy()
        tk = 1.0
        prev_obj = objective(beta)
        converged = False
        for it in range(1, max_iter + 1):
            grad = 2.0 * SW @ accel - 2.0 * StW @ z
            beta_new = project(accel - grad / L)
            tk_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
            accel = beta_new + ((tk - 1.0) / tk_new) * (beta_new - beta)

            if it >= 30 and it % 10 == 0:
                cur_obj = objective(beta_new)
                denom = max(abs(prev_obj), 1.0)
                if abs(prev_obj - cur_obj) / denom < tol:
                    beta = beta_new
                    converged = True
                    break
                prev_obj = cur_obj
            beta = beta_new
            tk = tk_new
        return beta, converged

    def _solve_slsqp(self, beta0, z, S, SW, StW, W, n_bottom, n_q,
                     lb, ub, explicit_node_bounds=False):
        """Scipy SLSQP fallback, used for signed-weight hierarchies."""
        from scipy.optimize import Bounds, LinearConstraint, minimize

        def unpack(v):
            return v.reshape(n_bottom, n_q)

        def obj(v):
            beta = unpack(v)
            diff = S @ beta - z
            val = float(np.einsum("nk,nm,mk->", diff, W, diff))
            grad = (2.0 * SW @ beta) - 2.0 * (StW @ z)
            return val, grad.reshape(-1)

        constraints = []
        scipy_bounds = None
        if n_q > 1:
            mono = self._monotonicity_matrix(n_bottom, n_q)
            constraints.append(
                LinearConstraint(mono, lb=0.0, ub=np.inf)
            )

        if explicit_node_bounds and (lb is not None or ub is not None):
            node_con = np.kron(S, np.eye(n_q))
            lb_vec = np.full(S.shape[0] * n_q, -np.inf)
            ub_vec = np.full(S.shape[0] * n_q, np.inf)
            if lb is not None:
                lb_vec[:] = lb
            if ub is not None:
                ub_vec[:] = ub
            constraints.append(
                LinearConstraint(node_con, lb=lb_vec, ub=ub_vec)
            )
        elif lb is not None or ub is not None:
            scipy_bounds = Bounds(
                np.full(n_bottom * n_q, lb if lb is not None else -np.inf),
                np.full(n_bottom * n_q, ub if ub is not None else np.inf),
            )

        result = minimize(
            obj,
            beta0.reshape(-1),
            jac=True,
            method="SLSQP",
            bounds=scipy_bounds,
            constraints=constraints,
            tol=1e-6,
            options={"maxiter": 1000, "ftol": 1e-6},
        )
        if not result.success:
            result = minimize(
                obj,
                beta0.reshape(-1),
                jac=True,
                method="SLSQP",
                bounds=scipy_bounds,
                constraints=constraints,
                tol=1e-4,
                options={"maxiter": 2000, "ftol": 1e-4},
            )
        if not result.success:
            raise ReconciliationError(
                "constrained reconciliation failed to converge: "
                f"{result.message}. No partial result is returned."
            )
        return unpack(result.x)

    # ------------------------------------------------------------------ #
    # post-conditions
    # ------------------------------------------------------------------ #
    def _assert_coherent(self, rec, time_index, rows=None):
        """Parent endpoint equals weighted sum of children endpoints.

        Coherence is checked through the explicit aggregation identity
        ``S S^+ y = y`` (projection onto the summation column space),
        which is the correct test when S has more rows than columns; a
        plain least-squares residual would conflate rank deficiency with a
        genuine violation.

        Parameters
        ----------
        rows : array-like of int, optional
            Subset of time rows to check (the QP-reconciled rows).
        """
        S = self._S
        projector = self._get_col_space_projector()
        check_rows = range(rec.shape[0]) if rows is None else rows
        for t in check_rows:
            for k in range(rec.shape[2]):
                resid = rec[t, :, k] - projector @ rec[t, :, k]
                if np.max(np.abs(resid)) > 1e-7:
                    raise ReconciliationError(
                        "internal coherence check failed at "
                        f"time={time_index[t]}, quantile index={k}; "
                        f"max residual={np.max(np.abs(resid)):.3e}"
                    )
        return True

    def _assert_no_crossing(self, rec, alphas):
        diffs = np.diff(rec, axis=-1)
        if np.min(diffs) < -1e-7:
            raise ReconciliationError(
                "internal quantile ordering check failed; min adjacent "
                f"difference={np.min(diffs):.3e}"
            )

    def _assert_bounds(self, rec):
        lb, ub = self.bounds
        if lb is not None and np.nanmin(rec) < lb - 1e-7:
            raise ReconciliationError(
                f"lower bound {lb} violated; min={np.nanmin(rec):.6g}"
            )
        if ub is not None and np.nanmax(rec) > ub + 1e-7:
            raise ReconciliationError(
                f"upper bound {ub} violated; max={np.nanmax(rec):.6g}"
            )


# ====================================================================== #
# helpers
# ====================================================================== #
def _mean_width(arr):
    """Mean interval width across nodes and time, for the outer quantiles."""
    if arr.shape[-1] < 2:
        return 0.0
    return float(np.mean(arr[..., -1] - arr[..., 0]))


def _isotonic_rows(beta):
    """Pool-adjacent-violators isotonic projection on every row."""
    out = np.empty_like(beta)
    for j in range(beta.shape[0]):
        out[j] = _pav(beta[j])
    return out


def _pav(y):
    """Pool-adjacent-violators: nearest non-decreasing sequence to y."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    vals, weights, idx = [], [], []
    for i in range(n):
        vals.append(y[i])
        weights.append(1)
        idx.append(i)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = weights[-2] + weights[-1]
            v = (
                vals[-2] * weights[-2] + vals[-1] * weights[-1]
            ) / w
            vals[-2:] = [v]
            weights[-2:] = [w]
            idx[-2:] = [idx[-2]]
    out = np.empty(n)
    for v, start, w in zip(vals, idx, weights):
        out[start:start + w] = v
    return out


def _shrink_covariance(X, sample_cov):
    """Ledoit-Wolf style diagonal shrinkage of the sample covariance."""
    n = X.shape[0]
    diag = np.diag(np.diag(sample_cov))
    Xc = X - X.mean(axis=0)
    # shrinkage intensity (Schafer-Strimmer style, as in sktime MinT)
    std = np.sqrt(np.diag(sample_cov))
    if np.any(std == 0):
        return sample_cov
    Z = Xc / std
    Z2 = Z**2
    scale = ((Z.T @ Z) ** 2) / n
    hovar = (Z2.T @ Z2) - scale
    hovar = (n / (n - 1) ** 3) * hovar
    np.fill_diagonal(hovar, 0.0)
    corr = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    denom = (corr**2).sum()
    if denom == 0:
        lamb = 0.0
    else:
        lamb = float(np.clip(hover.sum() / denom, 0.0, 1.0))
    return lamb * diag + (1.0 - lamb) * sample_cov


def _top_node(structure):
    return structure.level_nodes_[0][0]
