"""Check that ``folds.fold_data`` reproduces the upstream skrub pipeline.

Fits a short HGB median model through ``elf.make_data_op`` on one CV fold and
the same model on the arrays of ``folds.fold_data``, and compares the test
predictions, the test times and the number of folds.

    python scripts/check_folds.py --fold 0
"""

import argparse

import numpy as np

import electricity_load_forecasting as elf
import folds
from electricity_load_forecasting.quantile_regressor import HGBQuantileRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--max_iter", type=int, default=30)
args = parser.parse_args()

data_op = elf.make_data_op(
    horizons=folds.HORIZONS,
    model="hgb_quantile",
    quantiles=(0.5,),
    max_iter=args.max_iter,
)
env = elf.get_env() | {"city_weather_fetcher": elf.fetch_city_weather_forecast}
cv_splits = list(data_op.skb.iter_cv_splits(environment=env))
assert len(cv_splits) == folds.n_folds(), (len(cv_splits), folds.n_folds())
split = cv_splits[args.fold]
learner = data_op.skb.make_learner().fit(split["train"])
upstream = learner.predict(split["test"])

hgb_params = dict(
    random_state=0,
    max_iter=args.max_iter,
    early_stopping=True,
    n_iter_no_change=100,
    max_depth=None,
    learning_rate=0.1,
    max_leaf_nodes=31,
    min_samples_leaf=20,
)
for h in folds.HORIZONS:
    d = folds.fold_data(args.fold, h)
    assert (
        split["X_test"]["prediction_time"].to_numpy() == d["time_test"]
    ).all()
    ours = (
        HGBQuantileRegressor(quantiles=(0.5,), hgb_params=hgb_params)
        .fit(d["X_train"], d["y_train"])
        .predict(d["X_test"])["q_0.5"]
        .to_numpy()
    )
    ref = upstream[f"{h}h__q_0.5"].to_numpy()
    print(
        f"{h}h: n_train={d['X_train'].shape} n_test={d['X_test'].shape[0]} "
        f"max abs diff={np.abs(ours - ref).max():.3g}"
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-6)
print(f"OK: {folds.n_folds()} folds, fold {args.fold} matches the skrub pipeline")
