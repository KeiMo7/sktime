"""End-to-end acceptance scenario for the vintage module.

Reproduces the business problem from the spec: monthly indicators with
preliminary releases revised two months later, point-in-time backtest,
revision diagnostics, and the backward-compatibility guarantee.
"""
import numpy as np
import pandas as pd

from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.split import ExpandingWindowSplitter
from sktime.vintage import (
    VintageAwareSplitter,
    VintageFrame,
    revision_impact,
    revision_report,
    revision_summary,
    vintage_evaluate,
)

rng = np.random.default_rng(2024)
periods = pd.period_range("2022-01", periods=36, freq="M")

# two hierarchical series: "total" and "component"
rows = []
for series, scale in (("total", 100.0), ("component", 40.0)):
    truth = scale + rng.normal(scale=scale * 0.05, size=len(periods))
    for k, (p, v) in enumerate(zip(periods, truth)):
        # preliminary value at T+1, noisy
        prelim = v + rng.normal(scale=scale * 0.04)
        rows.append((series, p, p + 1, prelim))
        # revised final value at T+3, for "total" only (component never revised)
        if series == "total":
            rows.append((series, p, p + 3, v))

idx = pd.MultiIndex.from_tuples(
    [(s, p, pub) for s, p, pub, _ in rows],
    names=["series", "target_period", "published_at"],
)
vf = VintageFrame(
    pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
)
print(f"frame: {vf.n_series} series, {vf.n_periods} periods, "
      f"{len(vf)} published versions")

# 1. revision diagnostics
report = revision_report(vf)
summary = revision_summary(vf)
print("\nrevision summary:")
print(summary[["n_periods", "n_revised", "fraction_revised",
               "mean_abs_change", "max_abs_change"]])
assert summary.loc["total", "n_revised"] == len(periods)
assert summary.loc["component", "n_revised"] == 0

# 2. point-in-time backtest vs naive final-table backtest
cv_inner = ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=12, step_length=3)
vcv = VintageAwareSplitter(cv_inner)
f_naive = NaiveForecaster(strategy="mean", sp=12)
mape = MeanAbsolutePercentageError()

results = vintage_evaluate(f_naive, vf, vcv, scoring=mape)
print("\nvintage backtest (scored on preliminary releases):")
print(results[["cutoff", "as_of", "len_train_window",
               "test_MeanAbsolutePercentageError"]].head(4).to_string(index=False))

# 3. leakage assertion at every fold: training rows published <= as_of
for y_train, _, as_of in vcv.split_series(vf):
    assert (y_train.index.get_level_values(-1) <= as_of).all()
print("\nLEAKAGE ASSERTION: every fold uses only values published <= as_of")

# 4. revision impact: preliminary vs final scoring
impact = revision_impact(vf, f_naive, VintageAwareSplitter(cv_inner), scoring=mape)
print("\nrevision impact (preliminary vs final scoring), first rows:")
print(impact.head(3).to_string(index=False))

# 5. backward compatibility: plain snapshot path identical to evaluate
y_final = vf.latest().xs("total", level=0)["value"].rename("value")
old = evaluate(f_naive, cv_inner, y_final, scoring=mape, return_data=True)
vf_compat = VintageFrame.from_snapshot(y_final)
new = vintage_evaluate(
    f_naive, vf_compat, VintageAwareSplitter(cv_inner),
    scoring=mape, return_data=True,
)
pd.testing.assert_series_equal(
    old["test_MeanAbsolutePercentageError"].reset_index(drop=True),
    new["test_MeanAbsolutePercentageError"].reset_index(drop=True),
    check_names=False,
)
print("\nBACKWARD COMPAT: vintage path on plain snapshot == evaluate, bitwise")

# 6. parallel == sequential
par = vintage_evaluate(
    f_naive, vf, vcv, scoring=mape,
    backend="loky", backend_params={"n_jobs": 2},
)
drop_timing = lambda d: d.drop(columns=[c for c in d.columns if "time" in c])
pd.testing.assert_frame_equal(drop_timing(results), drop_timing(par))
print("PARALLEL: loky results identical to sequential")

print("\nALL ACCEPTANCE CHECKS PASSED")
