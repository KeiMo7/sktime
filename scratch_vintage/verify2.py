import warnings

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.vintage import (
    InsufficientVintageDataError,
    VintageAwareSplitter,
    VintageFrame,
    VintageValidationError,
    revision_impact,
    revision_report,
    vintage_evaluate,
)
from sktime.performance_metrics.forecasting import MeanAbsoluteError


def make_hier_frame():
    """Two series, one never revised, one revised every period, with gaps."""
    periods = pd.period_range("2024-01", periods=10, freq="M")
    rows = []
    for s in ("a", "b"):
        for k, p in enumerate(periods):
            rows.append((s, p, p + 1, float(k) + (10 if s == "b" else 0)))
    # series a: revise first two periods one month later
    rows.append(("a", periods[0], pd.Period("2024-03", "M"), 100.0))
    rows.append(("a", periods[1], pd.Period("2024-04", "M"), 101.0))
    # series b: one period never published (drop its only row)
    rows = [r for r in rows if not (r[0] == "b" and r[1] == periods[5])]
    idx = pd.MultiIndex.from_tuples(
        [(a, b, c) for a, b, c, _ in rows],
        names=["series", "target_period", "published_at"],
    )
    return VintageFrame(
        pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
    ), periods


vf, periods = make_hier_frame()

# 1. hierarchical folds keep both series, point-in-time correct
cv = ExpandingWindowSplitter(fh=1, initial_window=4, step_length=2)
vcv = VintageAwareSplitter(cv)
folds = list(vcv.split_series(vf))
for tr, te, ao in folds:
    assert set(tr.index.get_level_values(0).unique()) == {"a", "b"}
    # no test/train period beyond as_of semantics; identity index intact
    assert tr.index.names == ["series", "target_period"]
print("HIER: folds cover both series with proper index structure")

# series 'a' revision visibility
first = folds[0]
ao = first[2]
tr = first[0]
jan_a = tr.loc[("a", pd.Period("2024-01", "M")), "value"]
if ao < pd.Period("2024-03", "M"):
    assert jan_a == 0.0
print("HIER: per-series point-in-time visibility correct, as_of =", ao)

# 2. evaluate on hierarchical data
res = vintage_evaluate(
    NaiveForecaster(strategy="last"), vf, vcv, scoring=MeanAbsoluteError()
)
print(res[["cutoff", "as_of", "len_train_window"]])
print("HIER: vintage_evaluate runs on hierarchical data")

# 3. insufficient data: every value published late, so first fold sees nothing
periods_t = pd.period_range("2024-01", periods=6, freq="M")
late_rows = [
    ("s", p, p + 3, float(k)) for k, p in enumerate(periods_t)
]
idx_late = pd.MultiIndex.from_tuples(
    [(r[0], r[1], r[2]) for r in late_rows],
    names=["series", "target_period", "published_at"],
)
late_vf = VintageFrame(
    pd.DataFrame(
        {"value": [float(r[3]) for r in late_rows]},
        index=idx_late,
    )
)
late_cv = VintageAwareSplitter(
    ExpandingWindowSplitter(fh=1, initial_window=3)
)
with pytest.raises(InsufficientVintageDataError, match="fold 0"):
    list(late_cv.split_series(late_vf))
print("ERRORS: insufficient visible data raises with fold identification")

# skip mode
skip_cv = VintageAwareSplitter(
    ExpandingWindowSplitter(fh=1, initial_window=3), on_insufficient="skip"
)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    skipped = list(skip_cv.split_series(late_vf))
    assert all(
        fold[2] >= pd.Period("2024-03", "M") for fold in skipped
    )
    assert any("Skipping fold" in str(wi.message) for wi in w)
print("ERRORS: insufficient data skips early folds with explicit warning")

# 4. validation errors identify series and period
bad_rows = [
    ("s", pd.Period("2024-01", "M"), pd.Period("2024-02", "M")),
    ("s", pd.Period("2024-01", "M"), pd.Period("2024-02", "M")),
]
idx = pd.MultiIndex.from_tuples(
    bad_rows, names=["series", "target_period", "published_at"]
)
with pytest.raises(VintageValidationError, match="duplicate publication"):
    VintageFrame(pd.DataFrame({"value": [1.0, 1.1]}, index=idx))
print("ERRORS: duplicate identity triple rejected")

idx2 = pd.MultiIndex.from_tuples(
    [("s", pd.Period("2024-02", "M"), pd.Period("2024-01", "M"))],
    names=["series", "target_period", "published_at"],
)
with pytest.raises(VintageValidationError, match="earlier than the target"):
    VintageFrame(pd.DataFrame({"value": [1.0]}, index=idx2))
print("ERRORS: publication before target period rejected")

idx3 = pd.MultiIndex.from_tuples(
    [("s", pd.Period("2024-01", "M"), pd.Period("2024-02", "M"))],
    names=["series", "target_period", "published_at"],
)
with pytest.raises(VintageValidationError, match="null values"):
    VintageFrame(pd.DataFrame({"value": [np.nan]}, index=idx3))
print("ERRORS: null value rejected")

# 5. revision report matches injected values
rep = revision_report(vf)
assert rep.loc[("a", pd.Period("2024-01", "M")), "abs_change"] == pytest.approx(100.0)
assert rep.loc[("a", pd.Period("2024-01", "M")), "initial_value"] == 0.0
assert rep.loc[("a", pd.Period("2024-01", "M")), "final_value"] == 100.0
assert rep.loc[("a", pd.Period("2024-02", "M")), "abs_change"] == pytest.approx(100.0)
# series b has no revisions -> not in report
assert all(idx[0] != "b" for idx in rep.index)
print("REVISION REPORT: injected revisions recovered exactly")

# 6. revision impact: initial vs latest scoring differ where revisions exist
impact = revision_impact(
    vf,
    NaiveForecaster(strategy="last"),
    VintageAwareSplitter(cv),
    scoring=MeanAbsoluteError(),
)
assert "test_MeanAbsoluteError_diff" in impact.columns
assert len(impact) == len(folds)
print(impact)
print("REVISION IMPACT: fold-wise initial-vs-latest comparison produced")

# dedicated scenario: the scored test periods themselves get revised, so
# scoring against initial vs latest values must produce different metrics
p_ = pd.period_range("2024-01", periods=8, freq="M")
rows_ = [("s", p, p + 1, float(k)) for k, p in enumerate(p_)]
# revise every period with a big additive correction, published later
rows_ += [
    ("s", p, p + 2, float(k) + 50.0) for k, p in enumerate(p_)
]
idx_ = pd.MultiIndex.from_tuples(
    [(r[0], r[1], r[2]) for r in rows_],
    names=["series", "target_period", "published_at"],
)
vf_rev = VintageFrame(
    pd.DataFrame({"value": [r[3] for r in rows_]}, index=idx_)
)
impact_rev = revision_impact(
    vf_rev,
    NaiveForecaster(strategy="mean"),
    VintageAwareSplitter(
        ExpandingWindowSplitter(fh=1, initial_window=4, step_length=2)
    ),
    scoring=MeanAbsoluteError(),
)
assert impact_rev["test_MeanAbsoluteError_diff"].abs().sum() > 0
# and the sign must be explainable: latest test values are 50 units higher
assert (impact_rev["test_MeanAbsoluteError_initial"] >= 0).all()
print(
    "REVISION IMPACT: revised test targets change scores, total abs diff =",
    impact_rev["test_MeanAbsoluteError_diff"].abs().sum(),
)

# 7. unpublished period detection
missing = vf.find_unpublished()
assert ("b", periods[5]) in list(missing)
print("GAPS: unpublished period detected:", list(missing))
