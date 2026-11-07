"""Collect the teacher's test-fold predictions as a model of the study.

Writes ``<name>_all_folds.parquet`` (all extended folds, for ``calibrate.py``),
``<name>_cv_predictions.parquet`` and ``<name>_scores.csv`` (outer folds; the
fit + predict time is the context encoding plus the test predictions).

    python scripts/teacher_predictions.py --teacher_dir results/raw/teacher_fc \
        --output_dir results/study --name kumo_tabular_q7
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

import folds

parser = argparse.ArgumentParser()
parser.add_argument("--teacher_dir", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--name", required=True)
args = parser.parse_args()

fold_preds, rows, all_rows = {}, [], []
for fold in folds.fold_ids():
    total = 0.0
    for h in folds.HORIZONS:
        z = np.load(args.teacher_dir / f"fold{fold}_h{h}.npz")
        meta = json.loads(str(z["meta"]))
        d = folds.fold_data(fold, h)
        pred = z["q_test"].astype(np.float64)
        fold_preds[(fold, h)] = pred
        total += meta["timings"]["fit"] + meta["timings"]["predict_test"]
        all_rows.append(
            pl.DataFrame(
                {"fold": fold, "horizon": h, "prediction_time": d["time_test"], "y": d["y_test"]}
                | {f"q_{q}": pred[:, i] for i, q in enumerate(folds.QUANTILES)}
            )
        )
    if fold >= 0:
        rows.append({"split": fold, "fit_time": 0.0, "score_time": total})

args.output_dir.mkdir(parents=True, exist_ok=True)
pl.concat(all_rows).write_parquet(args.output_dir / f"{args.name}_all_folds.parquet")
folds.predictions_frame({k: v for k, v in fold_preds.items() if k[0] >= 0}).write_parquet(
    args.output_dir / f"{args.name}_cv_predictions.parquet"
)
pl.DataFrame(rows).write_csv(args.output_dir / f"{args.name}_scores.csv")
