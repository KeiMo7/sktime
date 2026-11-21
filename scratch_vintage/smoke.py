import numpy as np
import pandas as pd

from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.vintage import (
    VintageAwareSplitter,
    VintageFrame,
    revision_report,
    revision_summary,
    vintage_evaluate,
)

# monthly periods, published with 1-month lag; period 2024-01 revised in 2024-03
rows = []
periods = pd.period_range("2024-01", periods=8, freq="M")
vals = np.arange(1.0, 9.0)
for p, v in zip(periods, vals):
    rows.append(("s", p, p + 1, v))  # initial release at T+1
rows.append(("s", periods[0], pd.Period("2024-03", "M"), 1.5))  # revision
idx = pd.MultiIndex.from_tuples(
    [(r[0], r[1], r[2]) for r in rows],
    names=["series", "target_period", "published_at"],
)
vf = VintageFrame(
    pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
 )
print(vf)
print()

inner = ExpandingWindowSplitter(fh=1, initial_window=3, step_length=1)
vcv = VintageAwareSplitter(inner)
for tr, te, ao in vcv.split_series(vf):
    print(
        "as_of", ao,
        "| train", [(p, v) for p, v in tr["value"].items()],
        "| test", list(te["value"].items()),
    )

print()
res = vintage_evaluate(
    NaiveForecaster(strategy="last"), vf, vcv, return_data=True
)
print(res[["cutoff", "as_of"]])
print(res.iloc[0]["y_train"])
print()
print(revision_report(vf))
print()
print(revision_summary(vf))
