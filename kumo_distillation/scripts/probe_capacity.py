"""Median-only probe of a student on one fold/horizon.

Reports, for several HGB settings, the test MAE against the labels, the test
MAE against the teacher median (student-teacher gap) and the fit time.
``--syn_dir`` adds ``--ratio`` x n_train synthetic rows labelled by the
teacher of that directory (same group weighting as ``distill.py``).

    python scripts/probe_capacity.py --fold 0 --horizon 1 --lag_diffs --residual \
        --syn_dir ../results/raw/teacher_block --ratio 2
"""

import argparse
import itertools
import json
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
parser.add_argument("--syn_dir", type=Path, default=None)
parser.add_argument("--ratio", type=float, default=2)
parser.add_argument("--learning_rates", type=float, nargs="+", default=[0.1])
parser.add_argument("--leaves", type=int, nargs="+", default=[31])
parser.add_argument("--max_iter", type=int, default=3000)
parser.add_argument("--residual", action="store_true")
parser.add_argument("--labels", action="store_true", help="fit the labels (squared error)")
parser.add_argument("--lag_diffs", action="store_true")
args = parser.parse_args()

h = args.horizon
d = folds.fold_data(args.fold, h)
names = d["feature_names"]
z = np.load(args.teacher_dir / f"fold{args.fold}_h{h}.npz")
q_train = z["q_train"] if args.oof_dir is None else np.load(
    args.oof_dir / f"fold{args.fold}_h{h}.npz"
)["q_train"]
X = d["X_train"]
target = d["y_train"] if args.labels else q_train[:, 3].astype(float)
w = np.ones(len(target))
if args.syn_dir is not None:
    zs = np.load(args.syn_dir / f"fold{args.fold}_h{h}.npz")
    meta = json.loads(str(zs["meta"]))
    n_syn = int(round(args.ratio * len(target)))
    X_syn = folds.synthetic_queries(
        X, meta["ratio"], meta["cutmix_seed"], meta.get("generator", "cutmix"), names
    )[:n_syn]
    X = np.concatenate([X, X_syn])
    target = np.concatenate([target, zs["q_syn"][:n_syn, 3]])
    w = np.concatenate([np.full(len(w), 0.5 / len(w)), np.full(n_syn, 0.5 / n_syn)])
    w *= len(w) / w.sum()
base = folds.last_observed_load(X, names, h)
base_test = folds.last_observed_load(d["X_test"], names, h)
if not args.residual:
    base, base_test = 0 * base, 0 * base_test
X_test = d["X_test"]
if args.lag_diffs:
    X = folds.lag_difference_features(X, names, h)
    X_test = folds.lag_difference_features(X_test, names, h)
n_real = len(d["y_train"])
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
    ).fit(X, target - base, sample_weight=w)
    dt = time.perf_counter() - t0
    pred = est.predict(X_test) + base_test
    train_gap = np.abs(est.predict(X[:n_real]) + base[:n_real] - target[:n_real]).mean()
    print(
        f"lr={lr} leaves={leaves}: n_iter={est.n_iter_} fit {dt:.0f}s "
        f"test MAE {np.abs(pred - d['y_test']).mean():.1f} "
        f"|student-teacher| test {np.abs(pred - teacher_test).mean():.1f} "
        f"train fit gap {train_gap:.1f}",
        flush=True,
    )
