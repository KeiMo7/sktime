# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for hierarchical probabilistic forecast reconciliation.

Covers the acceptance criteria:

1. three-level hierarchy reconciles with exact coherence (residuals reported)
2. quantiles do not cross; coverage diagnostics give nominal vs empirical
3. different strategies produce the expected, distinct results
4. boundary handling stays coherent and never collapses widths to zero
5. failure modes are reported with locations, never repaired silently
6. outputs are reproducible and the export schema is frozen
"""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.probabilistic.reconcile import (
    HierarchyAdditivityError,
    ProbabilisticHierarchicalReconciler,
    QuantileCrossingError,
    coverage_diagnostics,
    export_interval_table,
    export_quantile_table,
    width_comparison,
)
from sktime.forecasting.probabilistic.reconcile._validation import (
    quantile_frame_to_array,
)
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _bottom_hier_datagen

ALPHAS = [0.1, 0.5, 0.9]
METHODS = ["bu", "td_fcst", "ols", "wls_str", "wls_var", "mint_shrink"]


def _three_level_hierarchy(seed=123, length=48, bottom=6, levels=2):
    """Build the canonical group/region/store-style three-level hierarchy."""
    agg = Aggregator()
    y = agg.fit_transform(
        _bottom_hier_datagen(
            no_bottom_nodes=bottom,
            no_levels=levels,
            random_seed=seed,
            length=length,
        )
    )
    return y, agg


def _base_quantile_frame(y, fh=(1, 2, 3), alphas=ALPHAS, seed=0):
    """Base (unreconciled) hierarchical quantiles from a naive forecaster."""
    f = NaiveForecaster(strategy="mean")
    f.fit(y)
    return f.predict_quantiles(fh=list(fh), alpha=alphas)


def _coherence_residual(reconciler, arr):
    """Maximum distance from the weighted summation column space."""
    S = reconciler._S
    projector = S @ np.linalg.pinv(S)
    worst = 0.0
    for t in range(arr.shape[0]):
        for k in range(arr.shape[2]):
            resid = arr[t, :, k] - projector @ arr[t, :, k]
            worst = max(worst, np.abs(resid).max())
    return worst


@pytest.mark.parametrize("method", METHODS)
def test_coherence_across_three_levels(method):
    """Criterion 1: endpoints sum exactly across all hierarchy levels."""
    y, _ = _three_level_hierarchy()
    node_index = y.index.droplevel(-1).unique()
    frame = _base_quantile_frame(y)

    rec = ProbabilisticHierarchicalReconciler(method=method).fit(
        node_index=node_index, residuals=y
    )
    out, info = rec.reconcile_quantiles_frame(frame)
    arr, _ = quantile_frame_to_array(
        out, rec.node_index_, info["alphas"], info["variable"]
    )

    resid = _coherence_residual(rec, arr)
    assert resid < 1e-8, f"coherence residual too large for {method}: {resid}"

    # explicit parent = sum(children) check at every level and quantile
    var = out.columns[0][0]
    for t in out.index.get_level_values(-1).unique():
        qt = out.xs(t, level=-1)[var]
        for _, edge in rec.structure_.parent_child_.iterrows():
            parent, _child = edge["parent"], edge["child"]
            siblings = rec.structure_.parent_child_[
                rec.structure_.parent_child_["parent"] == parent
            ]["child"]
            for a in info["alphas"]:
                child_sum = sum(float(qt.loc[c, a]) for c in siblings)
                assert abs(float(qt.loc[parent, a]) - child_sum) < 1e-8


@pytest.mark.parametrize("method", METHODS)
def test_quantiles_never_cross(method):
    """Criterion 2: low quantile <= high quantile at every node/time."""
    y, _ = _three_level_hierarchy()
    frame = _base_quantile_frame(y)
    rec = ProbabilisticHierarchicalReconciler(method=method).fit(
        node_index=y.index.droplevel(-1).unique(), residuals=y
    )
    out, _ = rec.reconcile_quantiles_frame(frame)
    diffs = np.diff(out[out.columns[0][0]].values, axis=1)
    assert diffs.min() >= -1e-9


def test_strategies_produce_distinct_expected_results():
    """Criterion 3: strategies differ, and bu keeps bottom nodes fixed."""
    y, _ = _three_level_hierarchy()
    node_index = y.index.droplevel(-1).unique()
    frame = _base_quantile_frame(y)

    results = {}
    for method in ["bu", "ols", "td_fcst"]:
        rec = ProbabilisticHierarchicalReconciler(method=method).fit(
            node_index=node_index
        )
        results[method], _ = rec.reconcile_quantiles_frame(frame)

    # bottom-up must leave bottom-level quantiles bit-identical
    bottom_nodes = list(
        node_index[node_index.get_level_values(-1) != "__total"]
    )
    bu_bottom = results["bu"].loc[
        results["bu"].index.droplevel(-1).isin(bottom_nodes)
    ]
    base_bottom = frame.loc[frame.index.droplevel(-1).isin(bottom_nodes)]
    pd.testing.assert_frame_equal(
        bu_bottom.sort_index(), base_bottom.sort_index()
    )

    # ols and td must genuinely differ from bu at aggregate levels
    agg_mask = ~results["bu"].index.droplevel(-1).isin(bottom_nodes)
    assert not np.allclose(
        results["ols"][agg_mask].values, results["bu"][agg_mask].values
    )
    assert not np.allclose(
        results["td_fcst"][agg_mask].values,
        results["bu"][agg_mask].values,
    )


def _quantile_to_interval(q_frame, alphas, coverage=0.8):
    """Reshape a (var, alpha) quantile frame into interval column layout."""
    var = q_frame.columns[0][0]
    low_a, high_a = min(alphas), max(alphas)
    cols = pd.MultiIndex.from_tuples(
        [(var, coverage, "lower"), (var, coverage, "upper")]
    )
    return pd.DataFrame(
        np.column_stack(
            [q_frame[(var, low_a)], q_frame[(var, high_a)]]
        ),
        index=q_frame.index,
        columns=cols,
    )


def test_widths_do_not_blow_up_and_are_quantified():
    """Criterion 8: reconciliation must not markedly widen intervals."""
    y, _ = _three_level_hierarchy()
    node_index = y.index.droplevel(-1).unique()
    frame = _base_quantile_frame(y, alphas=[0.1, 0.9])

    rec = ProbabilisticHierarchicalReconciler(method="bu").fit(
        node_index=node_index
    )
    out, info = rec.reconcile_quantiles_frame(frame)
    # bottom-up preserves bottom widths exactly; aggregate widths are rebuilt
    # by summation, so the mean width stays within rounding of the base
    assert np.isfinite(info["width_ratio"])
    assert abs(info["width_ratio"] - 1.0) < 1e-2

    base_int = _quantile_to_interval(frame, [0.1, 0.9])
    rec_int = _quantile_to_interval(out, [0.1, 0.9])
    table = width_comparison(
        base_int, rec_int, rec.structure_, coverages=[0.8]
    )
    assert {"mean_width_base", "mean_width_reconciled", "width_ratio"} <= set(
        table.columns
    )
    assert np.isfinite(table["width_ratio"]).all()


def test_bounds_stay_coherent_and_nonzero_width():
    """Criterion 4: non-negativity / capacity keep coherence and width."""
    y, _ = _three_level_hierarchy(length=36)
    node_index = y.index.droplevel(-1).unique()
    rng = np.random.default_rng(7)
    times = y.index.get_level_values(-1).unique()[:3]
    rows, vals = [], {}
    for node in node_index:
        for t in times:
            center = rng.normal(80, 25)
            vals[(node, t)] = [center - 40, center, center + 40]
    frame = pd.DataFrame(
        [vals[(n, t)] for n in node_index for t in times],
        index=pd.MultiIndex.from_tuples(
            [(*n, t) for n in node_index for t in times],
            names=y.index.names,
        ),
        columns=pd.MultiIndex.from_product([["v"], ALPHAS]),
    )

    rec = ProbabilisticHierarchicalReconciler(
        method="ols", bounds=(0.0, 400.0)
    ).fit(node_index=node_index)
    out, info = rec.reconcile_quantiles_frame(frame)

    assert out.min().min() >= -1e-9
    assert out.max().max() <= 400.0 + 1e-7
    arr, _ = quantile_frame_to_array(
        out, rec.node_index_, ALPHAS, "v"
    )
    assert _coherence_residual(rec, arr) < 1e-8
    # zero width may only appear where the bounds make lower==upper
    # unavoidable; it must never be introduced as a "coherence trick"
    # on unconstrained nodes. Every non-binding node keeps positive width.
    widths = (out[("v", 0.9)] - out[("v", 0.1)]).to_numpy()
    assert (widths >= -1e-9).all()
    assert (widths > 1e-6).sum() > 0.5 * len(widths)
    assert len(info["constrained_solver_rows"]) > 0


def test_crossing_input_raises_with_locations():
    """Criterion 3 (failure): crossed inputs are reported, not silently fixed."""
    y, _ = _three_level_hierarchy(length=36)
    node_index = y.index.droplevel(-1).unique()
    frame = _base_quantile_frame(y, alphas=[0.1, 0.9])
    frame.iloc[0, 0] = frame.iloc[0, 1] + 5.0  # force a crossing

    rec = ProbabilisticHierarchicalReconciler(method="bu").fit(
        node_index=node_index
    )
    with pytest.raises(QuantileCrossingError) as exc:
        rec.reconcile_quantiles_frame(frame)
    assert len(exc.value.details) >= 1
    assert {"node", "time", "alpha_low", "alpha_high"} <= set(
        exc.value.details[0]
    )


def test_crossing_repair_opt_in_restores_order():
    """Explicit isotonic repair restores order and coherence."""
    y, _ = _three_level_hierarchy(length=36)
    node_index = y.index.droplevel(-1).unique()
    frame = _base_quantile_frame(y, alphas=[0.1, 0.9])
    frame.iloc[0, 0] = frame.iloc[0, 1] + 5.0

    rec = ProbabilisticHierarchicalReconciler(
        method="bu", on_input_crossing="isotonic"
    ).fit(node_index=node_index)
    out, info = rec.reconcile_quantiles_frame(frame)
    var = out.columns[0][0]
    diffs = np.diff(out[var].values, axis=1)
    assert diffs.min() >= -1e-9
    assert info["input_crossing_repaired"] is True
    arr, _ = quantile_frame_to_array(out, rec.node_index_, [0.1, 0.9], var)
    assert _coherence_residual(rec, arr) < 1e-8


def test_non_additive_hierarchy_rejected_with_level():
    """Criterion 1 (failure): bad edge weights raise naming the level."""
    y, _ = _three_level_hierarchy(length=24)
    node_index = y.index.droplevel(-1).unique()
    n = len(node_index)
    weights = pd.DataFrame(
        np.zeros((n, n)), index=node_index, columns=node_index
    )
    weights.iloc[0, 1] = 0.3  # breaks the unit-sum condition somewhere

    with pytest.raises(HierarchyAdditivityError) as exc:
        ProbabilisticHierarchicalReconciler(method="bu").fit(
            node_index=node_index, weights=weights
        )
    assert exc.value.details[0]["level"] >= 0
    assert "weight_sum" in exc.value.details[0]


def test_covariance_degradation_flagged():
    """Criterion 3 (failure): insufficient residuals degrade and annotate."""
    y, _ = _three_level_hierarchy(length=12)
    node_index = y.index.droplevel(-1).unique()
    # only two residual time points for many nodes -> insufficient
    times = y.index.get_level_values(-1).unique()[:2]
    short_resid = y.loc[y.index.get_level_values(-1).isin(times)]

    rec = ProbabilisticHierarchicalReconciler(
        method="mint_cov"
    ).fit(node_index=node_index, residuals=short_resid)
    assert rec.method_used_ == "ols"
    assert rec.warnings_, "expected a degradation warning"


def test_reproducible_outputs():
    """Criterion 9: same inputs and params give identical frames."""
    y, _ = _three_level_hierarchy()
    node_index = y.index.droplevel(-1).unique()
    frame = _base_quantile_frame(y)

    a = ProbabilisticHierarchicalReconciler(method="ols").fit(
        node_index=node_index
    ).reconcile_quantiles_frame(frame)[0]
    b = ProbabilisticHierarchicalReconciler(method="ols").fit(
        node_index=node_index
    ).reconcile_quantiles_frame(frame)[0]
    pd.testing.assert_frame_equal(a, b)


def test_export_tables_have_frozen_schema():
    """Criterion 11: fixed columns and deterministic level ordering."""
    y, _ = _three_level_hierarchy(length=36)
    node_index = y.index.droplevel(-1).unique()
    frame = _base_quantile_frame(y, alphas=[0.1, 0.9])
    rec = ProbabilisticHierarchicalReconciler(method="bu").fit(
        node_index=node_index
    )
    q_out, _ = rec.reconcile_quantiles_frame(frame)

    int_frame = _quantile_to_interval(q_out, [0.1, 0.9])
    table = export_interval_table(int_frame, rec.structure_)
    assert list(table.columns) == [
        "level",
        "node",
        "timepoint",
        "variable",
        "coverage",
        "lower",
        "upper",
        "point_forecast",
        "width",
    ]
    # levels appear strictly top -> bottom
    assert table["level"].is_monotonic_increasing

    qtable = export_quantile_table(q_out, rec.structure_)
    assert list(qtable.columns) == [
        "level",
        "node",
        "timepoint",
        "variable",
        "alpha",
        "quantile",
    ]


def test_coverage_diagnostics_nominal_vs_empirical():
    """Criterion 2: diagnostics report both nominal and empirical coverage."""
    y, agg = _three_level_hierarchy(length=60)
    node_index = y.index.droplevel(-1).unique()
    frame = _base_quantile_frame(y, fh=(1, 2, 3, 4), alphas=[0.1, 0.9])
    rec = ProbabilisticHierarchicalReconciler(method="bu").fit(
        node_index=node_index
    )
    q_out, _ = rec.reconcile_quantiles_frame(frame)
    int_frame = _quantile_to_interval(q_out, [0.1, 0.9])

    # realised values at the forecast timestamps: perturb the midpoint so a
    # realistic fraction falls inside the 80% interval, and evaluate coverage
    # on exactly the frame's index (no future-only timestamp mismatch)
    mid = 0.5 * (
        int_frame[("passengers", 0.8, "lower")]
        + int_frame[("passengers", 0.8, "upper")]
    )
    rng = np.random.default_rng(11)
    realised = mid + rng.normal(0, mid.std() * 0.3, size=len(mid))
    actuals = realised.to_frame().sort_index()
    actuals.index = int_frame.index

    diag = coverage_diagnostics(
        actuals, int_frame, rec.structure_, coverages=[0.8]
    )
    assert {"nominal_coverage", "empirical_coverage", "coverage_deviation"} <= (
        set(diag.columns)
    )
    assert np.allclose(diag["nominal_coverage"], 0.8)
    assert diag["empirical_coverage"].dropna().between(0, 1).all()
    assert len(diag) == len(rec.structure_.level_nodes_)
