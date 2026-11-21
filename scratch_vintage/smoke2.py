import numpy as np
import pandas as pd

from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.vintage import (
    VintageAwareSplitter,
    VintageFrame,
    vintage_evaluate,
)

# publication at mid-month: Jan value published 2024-02-15,
# Jan revision published 2024-03-15
rows = [
    ("s", pd.Period("2024-01", "M"), pd.Timestamp("2024-02-15"), 1.0),
    ("s", pd.Period("2024-01", "M"), pd.Timestamp("2024-03-15"), 1.5),
    ("s", pd.Period("2024-02", "M"), pd.Timestamp("2024-03-15"), 2.0),
    ("s", pd.Period("2024-03", "M"), pd.Timestamp("2024-04-15"), 3.0),
    ("s", pd.Period("2024-04", "M"), pd.Timestamp("2024-05-15"), 4.0),
    ("s", pd.Period("2024-05", "M"), pd.Timestamp("2024-06-15"), 5.0),
    ("s", pd.Period("2024-06", "M"), pd.Timestamp("2024-07-15"), 6.0),
]
idx = pd.MultiIndex.from_tuples(
    [(a, b, c) for a, b, c, _ in rows],
    names=["series", "target_period", "published_at"],
)
vf = VintageFrame(
    pd.DataFrame({"value": [r[3] for r in rows]}, index=idx)
)

inner = ExpandingWindowSplitter(fh=1, initial_window=3, step_length=1)
vcv = VintageAwareSplitter(inner)
for tr, te, ao in vcv.split_series(vf):
    print(
        "as_of", ao,
        "| train", [(str(p), v) for p, v in tr["value"].items()],
        "| test", list(te["value"].items()),
    )

print()
res = vintage_evaluate(
    NaiveForecaster(strategy="last"), vf, vcv, return_data=True
)
print(res[["cutoff", "as_of", "len_train_window"]])
print()
print("y_train index type:", type(res.iloc[0]["y_train"].index), res.iloc[0]["y_train"].index)
print("y_test:", res.iloc[0]["y_test"])
print("y_pred:", res.iloc[0]["y_pred"])
