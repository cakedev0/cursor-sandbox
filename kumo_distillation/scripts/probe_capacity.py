"""Median-only probe of the student capacity on one fold/horizon.

Reports, for several HGB settings, the test MAE against the labels, the test
MAE against the teacher median (student-teacher gap) and the fit time.

    python scripts/probe_capacity.py --fold 0 --horizon 1 --oof_dir results/raw/teacher_oof4
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import folds

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--horizon", type=int, default=1)
parser.add_argument("--teacher_dir", type=Path, default=Path("../results/raw/teacher_fc"))
parser.add_argument("--oof_dir", type=Path, default=None)
parser.add_argument("--learning_rates", type=float, nargs="+", default=[0.1, 0.3])
parser.add_argument("--leaves", type=int, nargs="+", default=[31, 255])
parser.add_argument("--max_iter", type=int, default=3000)
parser.add_argument("--residual", action="store_true")
parser.add_argument("--labels", action="store_true", help="fit the labels (squared error)")
parser.add_argument("--lag_diffs", action="store_true")
args = parser.parse_args()

d = folds.fold_data(args.fold, args.horizon)
if args.lag_diffs:
    for k in ("X_train", "X_test"):
        d[k] = folds.lag_difference_features(d[k], d["feature_names"], args.horizon)
z = np.load(args.teacher_dir / f"fold{args.fold}_h{args.horizon}.npz")
q_train = z["q_train"] if args.oof_dir is None else np.load(
    args.oof_dir / f"fold{args.fold}_h{args.horizon}.npz"
)["q_train"]
target = d["y_train"] if args.labels else q_train[:, 3].astype(float)
base = folds.last_observed_load(d["X_train"], d["feature_names"], args.horizon)
base_test = folds.last_observed_load(d["X_test"], d["feature_names"], args.horizon)
if not args.residual:
    base, base_test = 0 * base, 0 * base_test
teacher_test = z["q_test"][:, 3]
print(f"teacher test MAE {np.abs(teacher_test - d['y_test']).mean():.1f}")
for lr, leaves in itertools.product(args.learning_rates, args.leaves):
    t0 = time.perf_counter()
    est = HistGradientBoostingRegressor(
        learning_rate=lr,
        max_leaf_nodes=leaves,
        min_samples_leaf=20,
        max_iter=args.max_iter,
        early_stopping=True,
        n_iter_no_change=100,
        random_state=0,
    ).fit(d["X_train"], target - base)
    dt = time.perf_counter() - t0
    pred = est.predict(d["X_test"]) + base_test
    train_gap = np.abs(est.predict(d["X_train"]) + base - target).mean()
    print(
        f"lr={lr} leaves={leaves}: n_iter={est.n_iter_} fit {dt:.0f}s "
        f"test MAE {np.abs(pred - d['y_test']).mean():.1f} "
        f"|student-teacher| test {np.abs(pred - teacher_test).mean():.1f} "
        f"train fit gap {train_gap:.1f}",
        flush=True,
    )
