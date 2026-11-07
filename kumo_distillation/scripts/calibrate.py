"""Post-hoc recalibration of any model from its past out-of-fold predictions.

The upstream ``hgb_quantile_recal`` model wraps the regressor in
``calibration.Recalibrator``: an inner walk-forward CV on the training set
(``TimeSeriesSplitter(min_train_days=365)``, last 4 inner folds) gives
out-of-fold predictions, from which a per-quantile shift (binned residual
quantile, smoothed by a spline ridge) is learned and added to the predictions.

For a distilled student, that inner CV would re-run the teacher 4 more times
per fold. Instead, this script uses the predictions of the walk-forward folds
that precede each outer fold, which ``distill.py`` already computes for the
extended folds (``folds.extended_splits``): outer fold k is recalibrated on
folds k-4 .. k-1, restricted to the rows before the 7-day train/test gap.
Each of these models was trained on data preceding its own test block only, so
no information from the test fold of k is used. The shift model is the
upstream ``Recalibrator._fit_shift_model`` (10 bins).

    python scripts/calibrate.py results/study/kd_hgb_cutmix2_all_folds.parquet
"""

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import polars as pl

import folds
from electricity_load_forecasting.calibration import Recalibrator

parser = argparse.ArgumentParser()
parser.add_argument("all_folds", type=Path, nargs="+")
parser.add_argument("--n_past_folds", type=int, default=4)
parser.add_argument("--n_bins", type=int, default=10)
parser.add_argument("--suffix", default="_recal")
parser.add_argument(
    "--method",
    choices=["spline", "constant"],
    default="spline",
    help="spline: upstream Recalibrator shift (a function of the predicted value); "
    "constant: one shift per quantile level (conformalized quantile regression)",
)
parser.add_argument(
    "--keep_median", action="store_true", help="do not shift the median prediction"
)
args = parser.parse_args()

GAP = datetime.timedelta(days=7)
QCOLS = [f"q_{q}" for q in folds.QUANTILES]


def recalibrate(df):
    shifter = Recalibrator(estimator=None, n_bins=args.n_bins)
    fold_preds, fit_times = {}, {}
    for fold in range(folds.n_folds()):
        fit_times[fold] = 0.0
        for h in folds.HORIZONS:
            test = df.filter((pl.col("fold") == fold) & (pl.col("horizon") == h))
            start = test["prediction_time"].min()
            past = df.filter(
                pl.col("fold").is_between(fold - args.n_past_folds, fold - 1)
                & (pl.col("horizon") == h)
                & (pl.col("prediction_time") < start - GAP)
            )
            t0 = time.perf_counter()
            corrected = test.select(QCOLS).to_numpy().astype(float, copy=True)
            y = past["y"].to_numpy()
            for i, (col, q) in enumerate(zip(QCOLS, folds.QUANTILES)):
                if args.keep_median and q == 0.5:
                    continue
                p = past[col].to_numpy()
                if args.method == "constant":
                    corrected[:, i] += np.quantile(y - p, q)
                    continue
                model, _ = shifter._fit_shift_model(p, y - p, q)
                corrected[:, i] += model.predict(corrected[:, [i]])
            fit_times[fold] += time.perf_counter() - t0
            fold_preds[(fold, h)] = np.sort(corrected, axis=1)
    return fold_preds, fit_times


for path in args.all_folds:
    name = path.name.removesuffix("_all_folds.parquet")
    df = pl.read_parquet(path).sort("fold", "horizon", "prediction_time")
    fold_preds, fit_times = recalibrate(df)
    out = path.parent / f"{name}{args.suffix}"
    folds.predictions_frame(fold_preds).write_parquet(f"{out}_cv_predictions.parquet")
    scores = path.parent / f"{name}_scores.csv"
    if scores.exists():
        base = pl.read_csv(scores)
        base.with_columns(
            fit_time=pl.col("fit_time") + pl.Series([fit_times[f] for f in base["split"]])
        ).write_csv(f"{out}_scores.csv")
    print(f"wrote {out}_cv_predictions.parquet")
