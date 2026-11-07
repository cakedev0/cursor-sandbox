"""Generate Kumo-Tabular teacher predictions for every fold and horizon.

For each fold of ``folds.extended_splits`` and each horizon, the teacher uses
the full training set of the fold as context (full-context supervision of
arXiv:2610.01435) and predicts the 7 quantiles on:

- the test rows (this is the ``kumo_tabular`` CV prediction),
- the training rows themselves (whose labels are in the context),
- ``--ratio`` x n_train synthetic rows (``folds.synthetic_queries``, CutMix by
  default, seeded by fold
  and horizon, so the student scripts can regenerate the same inputs; the
  first r x n_train rows are a valid sample for any ratio r <= ``--ratio``).

One ``.npz`` per fold/horizon is written to ``--output_dir``; existing files are
skipped, so the script can be resumed.

    python scripts/gen_teacher.py --output_dir results/raw/teacher_fc --ratio 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import folds
from teacher import KumoTeacher

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--ratio", type=float, default=4)
parser.add_argument("--folds", type=int, nargs="*", default=None)
parser.add_argument("--horizons", type=int, nargs="*", default=folds.HORIZONS)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--generator", choices=["cutmix", "block", "local"], default="cutmix")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)


def cutmix_seed(fold, horizon):
    return 1000 * (fold + 100) + horizon


fold_list = args.folds if args.folds is not None else folds.fold_ids()
# largest folds last: the outer folds are needed first for the main results
fold_list = sorted(fold_list, key=lambda f: (f < 0, f))
for fold in fold_list:
    for h in args.horizons:
        path = args.output_dir / f"fold{fold}_h{h}.npz"
        if path.exists():
            continue
        d = folds.fold_data(fold, h)
        X_syn = folds.synthetic_queries(
            d["X_train"], args.ratio, cutmix_seed(fold, h), args.generator, d["feature_names"]
        )
        teacher = KumoTeacher(batch_size=args.batch_size, seed=fold * 100 + h)
        timings = {}
        t0 = time.perf_counter()
        teacher.fit(d["X_train"], d["y_train"])
        torch.cuda.synchronize()
        timings["fit"] = time.perf_counter() - t0
        out = {}
        for name, X in [
            ("test", d["X_test"]),
            ("train", d["X_train"]),
            ("syn", X_syn),
        ]:
            t0 = time.perf_counter()
            out[f"q_{name}"] = teacher.predict(X).astype(np.float32)
            torch.cuda.synchronize()
            timings[f"predict_{name}"] = time.perf_counter() - t0
        teacher.clear()
        meta = {
            "fold": fold,
            "horizon": h,
            "n_train": int(d["X_train"].shape[0]),
            "n_test": int(d["X_test"].shape[0]),
            "n_features": int(d["X_train"].shape[1]),
            "ratio": args.ratio,
            "cutmix_seed": cutmix_seed(fold, h),
            "generator": args.generator,
            "quantiles": list(folds.QUANTILES),
            "timings": timings,
            "gpu": torch.cuda.get_device_name(),
        }
        np.savez(path, meta=json.dumps(meta), **out)
        mae = np.abs(out["q_test"][:, 3] - d["y_test"]).mean()
        print(json.dumps(meta | {"test_mae": float(mae)}), flush=True)
