"""Fit a family of HistGradientBoostingRegressor quantile models per fold.

With ``--supervision teacher`` (distillation), one
``HistGradientBoostingRegressor`` with the ``squared_error`` loss is fitted per
quantile level on the Kumo-Tabular teacher quantile predictions (no
ground-truth labels), for the real training rows and for ``--ratio`` x n_train
CutMix synthetic rows (arXiv:2610.01435). Real and synthetic rows get the same
total weight (the paper's sum of group-mean losses).

With ``--supervision labels``, the same family is fitted on the true labels
with the pinball (``quantile``) loss, as the upstream ``hgb_quantile`` model,
so that the target parametrizations can be compared without distillation.

Target parametrizations:

- ``--residual``: regress the target minus the load at prediction time (the
  ``lag_<h>`` feature), added back at prediction time;
- ``--target delta``: one model regresses the median, the others the offset of
  each quantile to the teacher median (teacher supervision only).

Predictions are written in the format of the upstream
``cross_validate_models.py`` (``<name>_cv_predictions.parquet`` for the outer
folds, ``<name>_scores.csv``), plus ``<name>_all_folds.parquet`` that also has
the earlier calibration-only folds, used by ``calibrate.py``.

    python scripts/distill.py --teacher_dir results/raw/teacher_fc \
        --output_dir results/study --ratio 2 --residual --name kd_hgb_resid_cutmix2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

import folds

parser = argparse.ArgumentParser()
parser.add_argument("--teacher_dir", type=Path, default=Path("results/raw/teacher_fc"))
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--name", required=True)
parser.add_argument(
    "--oof_dir",
    type=Path,
    default=None,
    help="take the targets of the real rows from these blocked out-of-fold teacher "
    "predictions (gen_teacher_oof.py) instead of the full-context ones",
)
parser.add_argument("--supervision", choices=["teacher", "labels"], default="teacher")
parser.add_argument("--ratio", type=float, default=2)
parser.add_argument("--residual", action="store_true")
parser.add_argument(
    "--lag_diffs",
    action="store_true",
    help="append the lag differences of folds.lag_difference_features",
)
parser.add_argument(
    "--wait", action="store_true", help="wait for teacher files still being generated"
)
parser.add_argument("--target", choices=["independent", "delta"], default="independent")
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--max_leaf_nodes", type=int, default=31)
parser.add_argument("--min_samples_leaf", type=int, default=20)
parser.add_argument("--max_iter", type=int, default=3000)
parser.add_argument("--n_iter_no_change", type=int, default=100)
parser.add_argument("--l2_regularization", type=float, default=0.0)
parser.add_argument("--folds", type=int, nargs="*", default=None)
parser.add_argument("--horizons", type=int, nargs="*", default=folds.HORIZONS)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
if args.supervision == "labels" and (args.ratio or args.target != "independent"):
    parser.error("--supervision labels requires --ratio 0 --target independent")

hgb_params = dict(
    random_state=0,
    max_iter=args.max_iter,
    early_stopping=True,
    n_iter_no_change=args.n_iter_no_change,
    max_depth=None,
    learning_rate=args.learning_rate,
    max_leaf_nodes=args.max_leaf_nodes,
    min_samples_leaf=args.min_samples_leaf,
    l2_regularization=args.l2_regularization,
)
MEDIAN = folds.QUANTILES.index(0.5)


def load_teacher(fold, h):
    path = args.teacher_dir / f"fold{fold}_h{h}.npz"
    while args.wait and not path.exists():
        time.sleep(60)
    z = np.load(path)
    return json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}


def training_set(d, h):
    """Return X, targets (n, n_quantiles), sample weights and teacher time."""
    n = d["X_train"].shape[0]
    if args.supervision == "labels":
        T = np.repeat(d["y_train"][:, None], len(folds.QUANTILES), axis=1)
        return d["X_train"], T, np.ones(n), 0.0
    meta, teacher = load_teacher(d["fold"], h)
    tt = meta["timings"]
    teacher_time = tt["fit"]
    q_train = teacher["q_train"]
    if args.oof_dir is None:
        teacher_time += tt["predict_train"]
    else:
        z = np.load(args.oof_dir / f"fold{d['fold']}_h{h}.npz")
        q_train = z["q_train"]
        oof_timings = json.loads(str(z["meta"]))["timings"]
        teacher_time += oof_timings["fit"] + oof_timings["predict_train"]
    n_syn = int(round(args.ratio * n))
    if not n_syn:
        teacher_time -= tt["fit"] if args.oof_dir is not None else 0.0
    X, T, w = [d["X_train"]], [q_train], [np.full(n, 1.0)]
    if n_syn:
        X_syn = folds.cutmix(d["X_train"], meta["ratio"], seed=meta["cutmix_seed"])
        assert X_syn.shape[0] >= n_syn, "teacher was run with a smaller ratio"
        X.append(X_syn[:n_syn])
        T.append(teacher["q_syn"][:n_syn])
        w = [np.full(n, 0.5 / n), np.full(n_syn, 0.5 / n_syn)]
    X, T, w = np.concatenate(X), np.concatenate(T).astype(np.float64), np.concatenate(w)
    teacher_time += tt["predict_syn"] * n_syn / max(len(teacher["q_syn"]), 1)
    return X, T, w * len(w) / w.sum(), teacher_time


def fit_predict(X, T, w, X_test, base, base_test):
    if args.residual:
        T = T - base[:, None]
    preds, n_iters, fit_dt, predict_dt = [], [], 0.0, 0.0
    for i, q in enumerate(folds.QUANTILES):
        target = T[:, i]
        if args.target == "delta" and i != MEDIAN:
            target = target - T[:, MEDIAN]
        if args.supervision == "labels":
            params = hgb_params | {"loss": "quantile", "quantile": q}
        else:
            params = hgb_params | {"loss": "squared_error"}
        t0 = time.perf_counter()
        est = HistGradientBoostingRegressor(**params).fit(X, target, sample_weight=w)
        t1 = time.perf_counter()
        preds.append(est.predict(X_test))
        predict_dt += time.perf_counter() - t1
        fit_dt += t1 - t0
        n_iters.append(int(est.n_iter_))
    preds = np.stack(preds, axis=1)
    if args.target == "delta":
        offsets = np.delete(np.arange(T.shape[1]), MEDIAN)
        preds[:, offsets] += preds[:, [MEDIAN]]
    if args.residual:
        preds += base_test[:, None]
    return np.sort(preds, axis=1), n_iters, fit_dt, predict_dt


fold_list = args.folds if args.folds is not None else folds.fold_ids()
# same order as the teacher generation (outer folds first)
fold_list = sorted(fold_list, key=lambda f: (f < 0, f))
fold_preds, rows, all_rows = {}, [], []
for fold in fold_list:
    fit_time = score_time = teacher_time = 0.0
    for h in args.horizons:
        d = folds.fold_data(fold, h) | {"fold": fold}
        X, T, w, teacher_dt = training_set(d, h)
        names = d["feature_names"]
        base = folds.last_observed_load(X, names, h)
        base_test = folds.last_observed_load(d["X_test"], names, h)
        X_test = d["X_test"]
        if args.lag_diffs:
            X = folds.lag_difference_features(X, names, h)
            X_test = folds.lag_difference_features(X_test, names, h)
        pred, n_iters, dt, predict_dt = fit_predict(X, T, w, X_test, base, base_test)
        teacher_time += teacher_dt
        fit_time += dt
        score_time += predict_dt
        fold_preds[(fold, h)] = pred
        mae = np.abs(pred[:, MEDIAN] - d["y_test"]).mean()
        print(
            f"{args.name} fold {fold} {h}h: n={X.shape[0]} student {dt:.1f}s "
            f"teacher {teacher_dt:.0f}s n_iter={n_iters} MAE {mae:.1f}",
            flush=True,
        )
        all_rows.append(
            pl.DataFrame(
                {
                    "fold": fold,
                    "horizon": h,
                    "prediction_time": d["time_test"],
                    "y": d["y_test"],
                }
                | {f"q_{q}": pred[:, i] for i, q in enumerate(folds.QUANTILES)}
            )
        )
    if fold >= 0:
        rows.append(
            {
                "split": fold,
                "fit_time": fit_time + teacher_time,
                "score_time": score_time,
                "student_fit_time": fit_time,
                "teacher_time": teacher_time,
            }
        )

pl.concat(all_rows).write_parquet(args.output_dir / f"{args.name}_all_folds.parquet")
if all(f in fold_list for f in range(folds.n_folds())) and len(args.horizons) == 3:
    folds.predictions_frame(fold_preds).write_parquet(
        args.output_dir / f"{args.name}_cv_predictions.parquet"
    )
    pl.DataFrame(rows).write_csv(args.output_dir / f"{args.name}_scores.csv")
(args.output_dir / f"{args.name}_config.json").write_text(
    json.dumps(vars(args) | {"hgb_params": hgb_params}, default=str, indent=2)
)
