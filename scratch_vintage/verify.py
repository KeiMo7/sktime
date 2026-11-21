import numpy as np
import pandas as pd

from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.split import ExpandingWindowSplitter
from sktime.vintage import (
    VintageAwareSplitter,
    VintageFrame,
    vintage_evaluate,
)

rng = np.random.default_rng(42)

# ------------------------------------------------------------------ #
# 1. backward compatibility: single snapshot, no version information
#    -> vintage_evaluate must agree bitwise with evaluate
# ------------------------------------------------------------------ #
y = pd.Series(
    rng.normal(size=24),
    index=pd.period_range("2024-01", periods=24, freq="M"),
    name="value",
)
cv = ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=10, step_length=2)
forecaster = NaiveForecaster(strategy="mean", sp=3)
metric = MeanAbsoluteError()

old = evaluate(forecaster, cv, y, scoring=metric, return_data=True)

vf = VintageFrame.from_snapshot(y)
vcv = VintageAwareSplitter(cv)
new = vintage_evaluate(
    forecaster, vf, vcv, scoring=metric, return_data=True
)

pd.testing.assert_series_equal(
    old["test_MeanAbsoluteError"].reset_index(drop=True),
    new["test_MeanAbsoluteError"].reset_index(drop=True),
    check_names=False,
)
pd.testing.assert_series_equal(
    old["cutoff"].reset_index(drop=True),
    new["cutoff"].reset_index(drop=True),
    check_names=False,
    check_dtype=False,
)
print("BACKWARD COMPAT: identical metric values and cutoffs")

# fold data content identical too
for i in range(len(old)):
    pd.testing.assert_series_equal(
        old.iloc[i]["y_train"], new.iloc[i]["y_train"], check_names=False
    )
    pd.testing.assert_series_equal(
        old.iloc[i]["y_test"], new.iloc[i]["y_test"], check_names=False
    )
print("BACKWARD COMPAT: identical fold contents")

# ------------------------------------------------------------------ #
# 2. parallel vs sequential reproducibility
# ------------------------------------------------------------------ #
seq = vintage_evaluate(forecaster, vf, vcv, scoring=metric, backend=None)
par = vintage_evaluate(
    forecaster, vf, vcv, scoring=metric, backend="threading",
    backend_params={"n_jobs": 2},
)
pd.testing.assert_frame_equal(
    seq.drop(columns=[c for c in seq.columns if "time" in c]),
    par.drop(columns=[c for c in par.columns if "time" in c]),
)
print("PARALLEL: threading identical to sequential")

par2 = vintage_evaluate(
    forecaster, vf, vcv, scoring=metric, backend="loky",
    backend_params={"n_jobs": 2},
)
pd.testing.assert_frame_equal(
    seq.drop(columns=[c for c in seq.columns if "time" in c]),
    par2.drop(columns=[c for c in par2.columns if "time" in c]),
)
print("PARALLEL: loky identical to sequential")

# repeatability
seq_again = vintage_evaluate(forecaster, vf, vcv, scoring=metric, backend=None)
pd.testing.assert_frame_equal(
    seq.drop(columns=[c for c in seq.columns if "time" in c]),
    seq_again.drop(columns=[c for c in seq_again.columns if "time" in c]),
)
print("REPEAT: identical on repeated runs")

# ------------------------------------------------------------------ #
# 3. leakage assertions on data with revisions + delays
# ------------------------------------------------------------------ #
periods = pd.period_range("2024-01", periods=12, freq="M")
rows = []
for k, p in enumerate(periods):
    # initial release at T+1 (mid-month timestamp), never revised
    rows.append(("s", p, (p + 1).to_timestamp(how="start") + pd.Timedelta(days=14),
                 float(k)))
# revise 2024-01 at publication instant 2024-05-15
rows.append(("s", periods[0], pd.Timestamp("2024-05-15"), 99.0))
idx = pd.MultiIndex.from_tuples(
    [(a, b, c) for a, b, c, _ in rows],
    names=["series", "target_period", "published_at"],
)
vf2 = VintageFrame(pd.DataFrame({"value": [r[3] for r in rows]}, index=idx))

cv2 = ExpandingWindowSplitter(fh=1, initial_window=4, step_length=1)
vcv2 = VintageAwareSplitter(cv2)
folds = list(vcv2.split_series(vf2))

# as long as the May revision is not visible, Jan value must be 0.0 everywhere
for tr, te, ao in folds:
    jan_val = tr["value"].droplevel(0).loc[pd.Period("2024-01", "M")]
    if ao <= pd.Period("2024-05", "M"):
        assert jan_val == 0.0, f"LEAK at as_of={ao}: Jan={jan_val}"
# and once visible, it must be the revised value
may_fold = [f for f in folds if f[2] == pd.Period("2024-05", "M")]
assert len(may_fold) == 1
# the May fold's cutoff is the start of May: mid-May revision still invisible
assert (
    may_fold[0][0]["value"].droplevel(0).loc[pd.Period("2024-01", "M")] == 0.0
)
june = [f for f in folds if f[2] == pd.Period("2024-06", "M")][0]
assert (
    june[0]["value"].droplevel(0).loc[pd.Period("2024-01", "M")] == 99.0
)
print("LEAKAGE: revision invisible before its publication, visible after")
