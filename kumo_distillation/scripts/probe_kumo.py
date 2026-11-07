"""Time Kumo-Tabular on one fold: upstream ``forward`` vs cached ``fit``/``predict``.

    python scripts/probe_kumo.py --fold 7 --horizon 1 --n_query 20000
"""

import argparse
import time

import numpy as np
import torch

import folds
from teacher import KumoTeacher, upstream_forward

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=7)
parser.add_argument("--horizon", type=int, default=1)
parser.add_argument("--n_query", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=8192)
args = parser.parse_args()

d = folds.fold_data(args.fold, args.horizon)
print("train", d["X_train"].shape, "test", d["X_test"].shape)

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
ref = upstream_forward(d["X_train"], d["y_train"], d["X_test"])
torch.cuda.synchronize()
print(f"upstream forward on test: {time.perf_counter() - t0:.1f}s "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
mae = np.abs(ref[:, 3] - d["y_test"]).mean()
print(f"  MAE median {mae:.1f}")

torch.cuda.reset_peak_memory_stats()
teacher = KumoTeacher(batch_size=args.batch_size)
t0 = time.perf_counter()
teacher.fit(d["X_train"], d["y_train"])
torch.cuda.synchronize()
print(f"cached fit: {time.perf_counter() - t0:.1f}s "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
t0 = time.perf_counter()
q_test = teacher.predict(d["X_test"])
torch.cuda.synchronize()
print(f"cached predict test: {time.perf_counter() - t0:.1f}s")
mae = np.abs(q_test[:, 3] - d["y_test"]).mean()
print(f"  MAE median {mae:.1f}; mean |cached - forward| median "
      f"{np.abs(q_test[:, 3] - ref[:, 3]).mean():.1f}")
X_syn = folds.cutmix(d["X_train"], args.n_query / d["X_train"].shape[0], seed=0)
t0 = time.perf_counter()
teacher.predict(X_syn)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"cached predict {X_syn.shape[0]} synthetic rows: {dt:.1f}s "
      f"({X_syn.shape[0] / dt:.0f} rows/s) "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
