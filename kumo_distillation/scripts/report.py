"""Metrics table and static figures for the models of a study directory.

Reads ``<name>_cv_predictions.parquet`` / ``<name>_scores.csv`` (upstream
``cross_validate_models.py`` format) and writes, into ``--output``:

- ``metrics.md`` / ``metrics.json``: pooled metrics over the 8 CV folds, with
  the upstream ``compare_predictions.compute_scores`` definitions (MAE / MAPE
  of the median, pinball-d2, coverage and width), plus the mean pinball loss;
- ``bars.png``: absolute metrics by horizon (mean over folds);
- ``differences.png``: paired differences to a baseline with the upstream
  moving-block bootstrap interval (``uncertainty.paired_difference``) and one
  dot per fold;
- ``coverage_width.png``, ``pareto.png``, ``reliability.png`` and
  ``timeseries.png``, the static counterparts of the upstream report figures.

    python scripts/report.py results/study --output journal/02-distillation \
        --models hgb_quantile_q7 kumo_tabular_q7 kd_hgb_cutmix2
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream/scripts/model_comparison"
sys.path.insert(0, str(UPSTREAM))
from compare_predictions import compute_scores  # noqa: E402
from uncertainty import paired_difference, row_index, row_loss  # noqa: E402

HORIZONS = (1, 12, 24)
LEVELS3 = (0.05, 0.5, 0.95)

parser = argparse.ArgumentParser()
parser.add_argument("study", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--models", nargs="+", required=True, help="models of the table")
parser.add_argument("--plot_models", nargs="+", default=None, help="subset for plots")
parser.add_argument("--baseline", default="hgb_quantile_q7")
parser.add_argument("--labels", type=Path, default=None, help="json name -> label")
parser.add_argument("--window", nargs=2, default=["2025-01-06", "2025-01-20"])
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
labels = json.loads(args.labels.read_text()) if args.labels else {}
plot_models = args.plot_models or args.models


def label(name):
    return labels.get(name, name)


frames = {
    m: pl.read_parquet(args.study / f"{m}_cv_predictions.parquet").sort("prediction_time")
    for m in args.models
}
ref = frames[args.baseline]
for m, df in frames.items():
    assert df.select("prediction_time", "split").equals(
        ref.select("prediction_time", "split")
    ), m
DAY, FOLD = row_index(ref)


def levels_of(df):
    return sorted(
        float(c.split("__q_")[1]) for c in df.columns if c.startswith("pred_1h__q_")
    )


def fit_predict_time(m):
    path = args.study / f"{m}_scores.csv"
    if not path.exists():
        return float("nan")
    s = pl.read_csv(path)
    return float(s["fit_time"].sum() + s["score_time"].sum())


def mean_pinball(df, h, levels):
    return float(np.mean([row_loss(df, "pinball", h, q).mean() for q in levels]))


rows = []
for m, df in frames.items():
    s = compute_scores(df)
    lv = levels_of(df)
    r = {"model": m}
    for h in HORIZONS:
        r[f"mae_{h}h"] = s[f"mae__{h}h"]
        r[f"mape_{h}h"] = s[f"mape__{h}h"]
        r[f"pinball3_{h}h"] = mean_pinball(df, h, LEVELS3)
        r[f"pinball7_{h}h"] = mean_pinball(df, h, lv) if len(lv) == 7 else float("nan")
        r[f"cov90_{h}h"] = s[f"coverage_90__{h}h"]
        r[f"width90_{h}h"] = s[f"width_90__{h}h"]
    for k in ("mae", "mape", "pinball3", "pinball7", "cov90", "width90"):
        r[f"{k}_avg"] = float(np.mean([r[f"{k}_{h}h"] for h in HORIZONS]))
    r["mae_avg"] = s["mae__average"]
    r["mape_avg"] = s["mape__average"]
    r["d2_pinball_avg"] = float(
        np.mean([s[f"d2_pinball_score__average__q_{q}"] for q in LEVELS3])
    )
    r["cov50_avg"] = s.get("coverage_50__average", float("nan"))
    r["cov80_avg"] = s.get("coverage_80__average", float("nan"))
    r["time_s"] = fit_predict_time(m)
    rows.append(r)
table = pl.DataFrame(rows)
(args.output / "metrics.json").write_text(json.dumps(rows, indent=1))


def fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return {"mw": f"{v:.0f}", "pct": f"{100 * v:.2f}", "cov": f"{v:.3f}", "d2": f"{v:.4f}", "s": f"{v:.0f}"}[kind]


cols = [
    ("MAE avg", "mae_avg", "mw"),
    ("MAE 1h", "mae_1h", "mw"),
    ("MAE 12h", "mae_12h", "mw"),
    ("MAE 24h", "mae_24h", "mw"),
    ("MAPE avg (%)", "mape_avg", "pct"),
    ("pinball 5/50/95 avg", "pinball3_avg", "mw"),
    ("pinball q7 avg", "pinball7_avg", "mw"),
    ("d2 pinball avg", "d2_pinball_avg", "d2"),
    ("cov 90 avg", "cov90_avg", "cov"),
    ("width 90 avg", "width90_avg", "mw"),
    ("cov 50 avg", "cov50_avg", "cov"),
    ("fit+predict (s)", "time_s", "s"),
]
lines = [
    "| model | " + " | ".join(c[0] for c in cols) + " |",
    "|---|" + "---:|" * len(cols),
]
for r in rows:
    lines.append(
        f"| {label(r['model'])} | " + " | ".join(fmt(r[k], kind) for _, k, kind in cols) + " |"
    )
(args.output / "metrics.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = colors + ["#444444", "#bbbb00", "#00aaaa", "#aa00aa"]
color = {m: colors[i % len(colors)] for i, m in enumerate(plot_models)}


def per_fold(df, kind, h, q=None):
    loss = row_loss(df, kind, h, q)
    return np.array([loss[FOLD == f].mean() for f in np.unique(FOLD)])


def fold_pinball(df, h):
    return np.mean([per_fold(df, "pinball", h, q) for q in LEVELS3], axis=0)


def fold_coverage(df, h):
    y = df[f"{h}h"].to_numpy()
    inside = (y >= df[f"pred_{h}h__q_0.05"].to_numpy()) & (
        y <= df[f"pred_{h}h__q_0.95"].to_numpy()
    )
    return np.array([inside[FOLD == f].mean() for f in np.unique(FOLD)])


# absolute metrics by horizon (mean over folds)
fig, axes = plt.subplots(1, 4, figsize=(18, 0.45 * len(plot_models) * 3 + 1.5), sharey=True)
metrics = [
    ("MAE (MW)", lambda df, h: per_fold(df, "mae", h).mean()),
    ("MAPE (%)", lambda df, h: per_fold(df, "mape", h).mean()),
    ("mean pinball 5/50/95 (MW)", lambda df, h: fold_pinball(df, h).mean()),
    ("90% coverage", lambda df, h: fold_coverage(df, h).mean()),
]
ypos = {}
for ax, (title, fn) in zip(axes, metrics):
    for j, h in enumerate(HORIZONS):
        for i, m in enumerate(plot_models):
            y = j * (len(plot_models) + 1) + i
            ypos[(h, m)] = y
            ax.barh(y, fn(frames[m], h), color=color[m], label=label(m) if j == 0 else None)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    if title.startswith("90%"):
        ax.axvline(0.9, color="gray", ls="--")
        ax.set_xlim(0.5, 1.0)
axes[0].set_yticks(
    [j * (len(plot_models) + 1) + (len(plot_models) - 1) / 2 for j in range(3)],
    [f"{h}h" for h in HORIZONS],
)
axes[0].invert_yaxis()
axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
fig.suptitle("Absolute metrics by horizon (mean over the 8 CV folds)")
fig.tight_layout()
fig.savefig(args.output / "bars.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# paired differences to the baseline
others = [m for m in plot_models if m != args.baseline]
kinds = [("mae", "MAE difference (MW)"), ("pinball", "mean pinball 5/50/95 difference (MW)"), ("width", "90% width difference (MW)")]
fig, axes = plt.subplots(1, 3, figsize=(18, 0.5 * len(others) * 3 + 1.5), sharey=True)
base = frames[args.baseline]
diff_json = {}
for ax, (kind, title) in zip(axes, kinds):
    for j, h in enumerate(HORIZONS):
        for i, m in enumerate(others):
            df = frames[m]
            if kind == "pinball":
                loss = np.mean([row_loss(df, "pinball", h, q) for q in LEVELS3], axis=0)
                bl = np.mean([row_loss(base, "pinball", h, q) for q in LEVELS3], axis=0)
            else:
                loss, bl = row_loss(df, kind, h), row_loss(base, kind, h)
            s = paired_difference(loss, bl, DAY, FOLD)
            diff_json[f"{m}|{kind}|{h}h"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in s.items()}
            y = j * (len(others) + 1) + i
            ax.errorbar(
                s["mean"], y, xerr=[[s["mean"] - s["low"]], [s["high"] - s["mean"]]],
                fmt="o", color=color[m], ms=7, capsize=3, label=label(m) if j == 0 else None,
            )
            ax.scatter(s["folds"], np.full(len(s["folds"]), y + 0.25), s=8, color=color[m], alpha=0.6)
    ax.axvline(0, color="gray", ls="--")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
axes[0].set_yticks(
    [j * (len(others) + 1) + (len(others) - 1) / 2 for j in range(3)],
    [f"{h}h" for h in HORIZONS],
)
axes[0].invert_yaxis()
axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
fig.suptitle(
    f"Paired differences vs {label(args.baseline)} (lower is better): pooled mean with 95% "
    "moving-block bootstrap interval (14-day blocks), small dots = per-fold means"
)
fig.tight_layout()
fig.savefig(args.output / "differences.png", dpi=110, bbox_inches="tight")
plt.close(fig)
(args.output / "differences.json").write_text(json.dumps(diff_json, indent=1))

# coverage vs width
fig, ax = plt.subplots(figsize=(8, 6))
markers = {1: "o", 12: "s", 24: "^"}
for m in plot_models:
    for h in HORIZONS:
        df = frames[m]
        ax.scatter(
            fold_coverage(df, h).mean(), per_fold(df, "width", h).mean(),
            color=color[m], marker=markers[h], s=60,
            label=label(m) if h == 1 else None,
        )
ax.axvline(0.9, color="gray", ls="--")
for h, mk in markers.items():
    ax.scatter([], [], marker=mk, color="k", label=f"{h}h")
ax.set_xlabel("90% interval coverage (mean over folds)")
ax.set_ylabel("90% interval width (MW, mean over folds)")
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
ax.grid(alpha=0.3)
ax.set_title("Coverage vs interval width")
fig.savefig(args.output / "coverage_width.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# pareto: accuracy vs fit + predict time
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
by_model = {r["model"]: r for r in rows}
for ax, (key, title) in zip(axes, [("mape_avg", "average MAPE (%)"), ("pinball3_avg", "average pinball 5/50/95 (MW)")]):
    for m in plot_models:
        r = by_model[m]
        v = r[key] * (100 if key == "mape_avg" else 1)
        ax.scatter(r["time_s"], v, color=color[m], s=60, label=label(m))
        ax.annotate(label(m), (r["time_s"], v), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("total fit + predict time over the 8 folds (s, log scale)")
    ax.set_ylabel(title)
    ax.grid(alpha=0.3, which="both")
fig.suptitle("Accuracy vs fit + predict time")
fig.tight_layout()
fig.savefig(args.output / "pareto.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# reliability: empirical P(y <= q_tau) vs tau
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
for ax, h in zip(axes, HORIZONS):
    ax.plot([0, 1], [0, 1], color="gray", ls="--")
    for m in plot_models:
        df = frames[m]
        lv = levels_of(df)
        y = df[f"{h}h"].to_numpy()
        emp = [(y <= df[f"pred_{h}h__q_{q}"].to_numpy()).mean() for q in lv]
        ax.plot(lv, emp, "o-", color=color[m], label=label(m), ms=4)
    ax.set_title(f"{h}h horizon")
    ax.set_xlabel("nominal quantile level")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("empirical P(y <= predicted quantile)")
axes[-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle("Reliability (marginal coverage curve), all CV test folds pooled")
fig.tight_layout()
fig.savefig(args.output / "reliability.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# true vs predicted over a short window
start, end = (np.datetime64(w) for w in args.window)
fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(16, 3.4 * len(HORIZONS)), sharex=True)
for ax, h in zip(axes, HORIZONS):
    first = True
    for m in plot_models:
        df = frames[m].with_columns(target_time=pl.col("prediction_time") + pl.duration(hours=h))
        t = df["target_time"].dt.replace_time_zone(None).to_numpy()
        sel = (t >= start) & (t < end)
        if first:
            ax.plot(t[sel], df[f"{h}h"].to_numpy()[sel], color="k", lw=1.5, label="actual load")
            first = False
        ax.plot(t[sel], df[f"pred_{h}h__q_0.5"].to_numpy()[sel], color=color[m], lw=1, label=label(m))
        ax.fill_between(
            t[sel], df[f"pred_{h}h__q_0.05"].to_numpy()[sel], df[f"pred_{h}h__q_0.95"].to_numpy()[sel],
            color=color[m], alpha=0.12,
        )
    ax.set_title(f"{h}h horizon: median and 5%-95% band")
    ax.set_ylabel("load (MW)")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(args.output / "timeseries.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"wrote figures to {args.output}")
