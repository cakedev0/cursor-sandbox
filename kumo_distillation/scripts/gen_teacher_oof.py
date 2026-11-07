"""Blocked out-of-fold Kumo-Tabular predictions on the training rows.

With the full training set as context (``gen_teacher.py``), the teacher sees
the label of each training row and of its neighbours one hour apart, and
essentially returns the label. At test time the closest context rows are at
least 7 days away. Here the training rows of each fold are cut into
``--n_blocks`` contiguous time blocks; each block is predicted with a context
that excludes the block and ``--gap_days`` on each side of it, which mimics the
test-time situation.

    python scripts/gen_teacher_oof.py --output_dir results/raw/teacher_oof
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch

import folds
from teacher import KumoTeacher

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--gap_days", type=float, default=7)
parser.add_argument("--n_estimators", type=int, default=8)
parser.add_argument("--folds", type=int, nargs="*", default=None)
parser.add_argument("--horizons", type=int, nargs="*", default=folds.HORIZONS)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

fold_list = args.folds if args.folds is not None else folds.fold_ids()
fold_list = sorted(fold_list, key=lambda f: (f < 0, f))
gap = np.timedelta64(int(args.gap_days * 24 * 3600), "s")
for fold in fold_list:
    for h in args.horizons:
        path = args.output_dir / f"fold{fold}_h{h}.npz"
        if path.exists():
            continue
        d = folds.fold_data(fold, h)
        times = d["time_train"].astype("datetime64[s]")
        blocks = np.array_split(np.arange(len(times)), args.n_blocks)
        q_oof = np.empty((len(times), len(folds.QUANTILES)), dtype=np.float32)
        timings = {"fit": 0.0, "predict_train": 0.0}
        for k, block in enumerate(blocks):
            start, end = times[block[0]] - gap, times[block[-1]] + gap
            context = (times < start) | (times > end)
            teacher = KumoTeacher(n_estimators=args.n_estimators, seed=fold * 100 + h + k)
            t0 = time.perf_counter()
            teacher.fit(d["X_train"][context], d["y_train"][context])
            torch.cuda.synchronize()
            timings["fit"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            q_oof[block] = teacher.predict(d["X_train"][block])
            torch.cuda.synchronize()
            timings["predict_train"] += time.perf_counter() - t0
            teacher.clear()
        meta = {
            "fold": fold,
            "horizon": h,
            "n_train": len(times),
            "n_blocks": args.n_blocks,
            "gap_days": args.gap_days,
            "n_estimators": args.n_estimators,
            "timings": timings,
            "date": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        np.savez(path, meta=json.dumps(meta), q_train=q_oof)
        mae = np.abs(q_oof[:, 3] - d["y_train"]).mean()
        cov = ((d["y_train"] >= q_oof[:, 0]) & (d["y_train"] <= q_oof[:, -1])).mean()
        print(json.dumps(meta | {"train_oof_mae": float(mae), "cov90": float(cov)}), flush=True)
