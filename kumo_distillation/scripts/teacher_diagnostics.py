"""Summarize what the Kumo-Tabular teacher predicts on real and synthetic queries.

For each fold and horizon, compares the teacher quantiles on the test rows,
on its own (in-context) train rows, on out-of-fold train rows when available
and on the synthetic queries: median MAE against the labels, mean 90%
interval width and 90% coverage. Writes a JSON table and a width plot.

    python scripts/teacher_diagnostics.py --out ../journal/02-distillation
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import folds

ROOT = Path(__file__).resolve().parent.parent / "results" / "raw"
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()


def summary(q, y=None):
    row = {"width90": float(np.mean(q[:, 6] - q[:, 0])), "n": int(len(q))}
    if y is not None:
        row["mae"] = float(np.mean(np.abs(q[:, 3] - y)))
        row["coverage90"] = float(np.mean((y >= q[:, 0]) & (y <= q[:, 6])))
    return row


rows = []
for generator, folds_ in [("cutmix", range(folds.n_folds())), ("block", [0]), ("local", [0])]:
    teacher_dir = ROOT / ("teacher_fc" if generator == "cutmix" else f"teacher_{generator}")
    for fold in folds_:
        for h in (1, 12, 24):
            path = teacher_dir / f"fold{fold}_h{h}.npz"
            if not path.exists():
                continue
            z = np.load(path)
            d = folds.fold_data(fold, h)
            row = {"generator": generator, "fold": fold, "horizon": h}
            row["test"] = summary(z["q_test"], d["y_test"])
            row["train_in_context"] = summary(z["q_train"], d["y_train"])
            row["synthetic"] = summary(z["q_syn"])
            oof = ROOT / "teacher_oof4" / f"fold{fold}_h{h}.npz"
            if oof.exists():
                row["train_out_of_fold"] = summary(np.load(oof)["q_train"], d["y_train"])
            rows.append(row)
            print(json.dumps(row), flush=True)

args.out.mkdir(parents=True, exist_ok=True)
(args.out / "teacher_diagnostics.json").write_text(json.dumps(rows, indent=1))

cut = [r for r in rows if r["generator"] == "cutmix"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, h in zip(axes, (1, 12, 24)):
    sel = [r for r in cut if r["horizon"] == h]
    x = np.array([r["fold"] for r in sel])
    for key, label in [
        ("test", "test rows"),
        ("train_in_context", "train rows (in context)"),
        ("train_out_of_fold", "train rows (out of fold)"),
        ("synthetic", "CutMix queries"),
    ]:
        if any(key not in r for r in sel):
            continue
        ax.plot(x, [r[key]["width90"] for r in sel], "o-", label=label)
    ax.set_yscale("log")
    ax.set_title(f"horizon {h}h")
    ax.set_xlabel("fold")
axes[0].set_ylabel("teacher mean 90% interval width (MW)")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(args.out / "teacher_width.png", dpi=120)
fig.savefig(args.out / "teacher_width.svg")
