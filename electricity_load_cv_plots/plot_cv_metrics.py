"""Plot the published walk-forward CV comparison.

Source table:
https://probabl-ai.github.io/skore-example-electricity-load/model-comparison/cv_comparison/report.html

Run from this directory with ``pixi run plot``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cv_metrics.json"
JOURNAL = ROOT / "journal" / "01-metric-plots"
ARTIFACTS = Path("/opt/cursor/artifacts")

HORIZONS = ("1h", "12h", "24h")
HORIZON_COLORS = {
    "1h": "#0072B2",
    "12h": "#E69F00",
    "24h": "#009E73",
}
QUANTILES = ("q_0.05", "q_0.95")
QUANTILE_COLORS = {"q_0.05": "#0072B2", "q_0.95": "#D55E00"}
QUANTILE_LABELS = {"q_0.05": "q = 0.05", "q_0.95": "q = 0.95"}

# Categorical colors and markers used only on the width-coverage scatters.
MODEL_COLORS = {
    "chronos2_ctx4000": "#0072B2",
    "hgb_quantile_q7": "#E69F00",
    "hgb_quantile_recal_q7": "#009E73",
    "kumo_tabular": "#D55E00",
    "tabicl_n8_full": "#CC79A7",
}
MODEL_MARKERS = {
    "chronos2_ctx4000": "o",
    "hgb_quantile_q7": "s",
    "hgb_quantile_recal_q7": "D",
    "kumo_tabular": "^",
    "tabicl_n8_full": "P",
}

# Two-line tick labels keep the published ids readable under seven groups.
MODEL_TICKS = {
    "chronos2_ctx4000": "chronos2\nctx4000",
    "hgb_mean": "hgb_mean",
    "hgb_quantile_q7": "hgb_quantile\nq7",
    "hgb_quantile_recal_q7": "hgb_quantile\nrecal_q7",
    "kumo": "kumo",
    "kumo_tabular": "kumo\ntabular",
    "tabicl_n8_full": "tabicl\nn8_full",
}

INTERVAL_MODELS = (
    "chronos2_ctx4000",
    "hgb_quantile_q7",
    "hgb_quantile_recal_q7",
    "kumo_tabular",
    "tabicl_n8_full",
)

# (dx points, dy points, horizontal alignment, vertical alignment)
_LABEL_CANDIDATES = (
    (8, 7, "left", "bottom"),
    (8, -7, "left", "top"),
    (-8, 7, "right", "bottom"),
    (-8, -7, "right", "top"),
    (12, 0, "left", "center"),
    (-12, 0, "right", "center"),
    (0, 10, "center", "bottom"),
    (0, -10, "center", "top"),
    (16, 12, "left", "bottom"),
    (16, -12, "left", "top"),
    (-16, 12, "right", "bottom"),
    (-16, -12, "right", "top"),
    (22, 4, "left", "center"),
    (-22, 4, "right", "center"),
    (6, 16, "left", "bottom"),
    (-6, -16, "right", "top"),
)


def load_metrics() -> dict:
    data = json.loads(DATA_PATH.read_text())
    models = data["models"]
    if models[0] != "chronos2_ctx4000" or "kumo_tabular" not in models:
        raise SystemExit(f"unexpected model list: {models}")
    coverage = data["metrics"]["coverage_90__1h"]
    present = tuple(model for model in models if coverage[model] is not None)
    if present != INTERVAL_MODELS:
        raise SystemExit(f"unexpected quantile models: {present}")
    # Guard against a swapped column. These are the 1h and 24h figures as published on 2026-11-02.
    if data["metrics"]["mae__1h"]["kumo_tabular"] != 276.9:
        raise SystemExit("MAE table does not match the published report")
    if data["metrics"]["d2_pinball_score__24h__q_0.05"]["hgb_quantile_q7"] != 0.7956:
        raise SystemExit("pinball D² table does not match the published report")
    return data


def value(data: dict, metric: str, model: str) -> float:
    raw = data["metrics"][metric][model]
    if raw is None:
        raise KeyError(f"{model} has no {metric}")
    return float(raw)


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.axisbelow": True,
            "svg.fonttype": "path",
        }
    )


def _thousands(number: float, _pos: int) -> str:
    return f"{number:,.0f}"


def _save(fig: plt.Figure, stem: str) -> None:
    JOURNAL.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    svg_path = JOURNAL / f"{stem}.svg"
    png_path = ARTIFACTS / f"{stem}.png"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(png_path, dpi=140, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _format_mae(item: float) -> str:
    if abs(item - round(item)) < 1e-6:
        return f"{item:.0f}"
    return f"{item:.1f}"


def plot_grouped_error(
    data: dict,
    metric_prefix: str,
    ylabel: str,
    title: str,
    stem: str,
    scale: float,
    formatter,
) -> None:
    models = data["models"]
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    n_series = len(HORIZONS)
    width = 0.78 / n_series
    x = np.arange(len(models))
    maxima = []
    for index, horizon in enumerate(HORIZONS):
        offset = (index - (n_series - 1) / 2) * width
        raw = [value(data, f"{metric_prefix}__{horizon}", model) * scale for model in models]
        maxima.extend(raw)
        bars = ax.bar(
            x + offset,
            raw,
            width=width * 0.92,
            label=horizon,
            color=HORIZON_COLORS[horizon],
            zorder=2,
            edgecolor="white",
            linewidth=0.5,
        )
        labels = [formatter(item) for item in raw]
        ax.bar_label(
            bars,
            labels=labels,
            padding=2,
            fontsize=7.5,
            rotation=90,
            color="#1c1915",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_TICKS[model] for model in models])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=18)
    # Rotated value labels sit above the bars. The tallest bar needs extra
    # headroom or its label is clipped by the axes.
    ax.set_ylim(0, max(maxima) * 1.36)
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands if scale == 1 else lambda v, _p: f"{v:.1f}"))
    ax.yaxis.grid(True, color="#e4ddd0", linewidth=0.7)
    ax.xaxis.grid(False)
    ax.legend(
        title="Horizon",
        ncol=3,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.012),
        borderaxespad=0.0,
    )
    ax.margins(x=0.02)
    _save(fig, stem)


def _label_boxes(points, sizes, combo, dpi):
    boxes = []
    for (px, py), (width, height), (dx, dy, ha, va) in zip(points, sizes, combo):
        ox = dx * dpi / 72
        oy = dy * dpi / 72
        anchor_x = px + ox
        anchor_y = py + oy
        if ha == "left":
            left = anchor_x
        elif ha == "right":
            left = anchor_x - width
        else:
            left = anchor_x - width / 2
        if va == "bottom":
            bottom = anchor_y
        elif va == "top":
            bottom = anchor_y - height
        else:
            bottom = anchor_y - height / 2
        boxes.append((left, bottom, left + width, bottom + height))
    return boxes


def _overlap(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _place_labels(ax: plt.Axes, labeled_points: list[tuple[float, float, str, str]]) -> None:
    """Pick non-overlapping label offsets by scoring every candidate combination."""
    fig = ax.figure
    probe = [
        ax.text(0, 0, text, fontsize=8.5)
        for _x, _y, text, _color in labeled_points
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sizes = []
    for text in probe:
        extent = text.get_window_extent(renderer)
        sizes.append((extent.width, extent.height))
        text.remove()

    display = [ax.transData.transform((x, y)) for x, y, _text, _color in labeled_points]
    nominal_x = ax.transData.transform((0.90, 0))[0]
    axes_box = ax.get_window_extent(renderer)
    axes_bounds = (axes_box.x0 + 2, axes_box.y0 + 2, axes_box.x1 - 2, axes_box.y1 - 2)
    dpi = fig.dpi
    marker = 9.0

    def score_one(box, cand, placed) -> float:
        dx, dy, _ha, _va = cand
        penalty = 0.02 * (dx * dx + dy * dy) + 3.0 * _outside(box, axes_bounds)
        penalty += 5.0 * _overlap(box, (nominal_x - 3.0, axes_box.y0, nominal_x + 3.0, axes_box.y1))
        for qx, qy in display:
            penalty += 10.0 * _overlap(box, (qx - marker, qy - marker, qx + marker, qy + marker))
        for other in placed:
            penalty += 12.0 * _overlap(box, other)
        return penalty

    # Place the most crowded points first so they keep the open space.
    remaining = set(range(len(labeled_points)))
    chosen = [None] * len(labeled_points)
    placed_boxes = []
    while remaining:
        def isolation(index: int) -> float:
            px, py = display[index]
            others = [display[j] for j in remaining if j != index]
            if not others:
                return 0.0
            return min((px - qx) ** 2 + (py - qy) ** 2 for qx, qy in others)

        current = min(remaining, key=isolation)
        best = None
        for cand in _LABEL_CANDIDATES:
            box = _label_boxes([display[current]], [sizes[current]], [cand], dpi)[0]
            penalty = score_one(box, cand, placed_boxes)
            if best is None or penalty < best[0]:
                best = (penalty, cand, box)
        chosen[current] = best[1]
        placed_boxes.append(best[2])
        remaining.remove(current)

    for (x, y, text, color), (dx, dy, ha, va) in zip(labeled_points, chosen):
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8.5,
            color=color,
            zorder=5,
        )


def _outside(box, bounds) -> float:
    """Area of ``box`` that falls outside ``bounds``."""
    inside = (
        max(box[0], bounds[0]),
        max(box[1], bounds[1]),
        min(box[2], bounds[2]),
        min(box[3], bounds[3]),
    )
    full = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if inside[2] <= inside[0] or inside[3] <= inside[1]:
        return full
    kept = (inside[2] - inside[0]) * (inside[3] - inside[1])
    return full - kept


def plot_width_coverage(data: dict, horizon: str) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.15))
    labeled = []
    widths = []
    for model in INTERVAL_MODELS:
        coverage = value(data, f"coverage_90__{horizon}", model)
        width = value(data, f"width_90__{horizon}", model)
        widths.append(width)
        ax.scatter(
            coverage,
            width,
            s=64,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            zorder=4,
            linewidth=0.7,
            edgecolor="white",
        )
        labeled.append((coverage, width, model, MODEL_COLORS[model]))
    ax.axvline(0.90, color="#222222", linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    span = max(widths) - min(widths)
    pad = max(span * 0.22, 80)
    ax.set_xlim(0.72, 1.015)
    ax.set_ylim(min(widths) - pad, max(widths) + pad * 1.15)
    ax.set_xlabel("Empirical coverage of the 90% interval")
    ax.set_ylabel("90% prediction interval width")
    ax.set_title(f"{horizon} horizon", loc="left")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.grid(True, color="#e4ddd0", linewidth=0.7)
    ax.annotate(
        "nominal 90%",
        xy=(0.90, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#222222",
        annotation_clip=False,
    )
    fig.tight_layout()
    _place_labels(ax, labeled)
    _save(fig, f"width_vs_coverage_{horizon}")


def plot_d2(data: dict) -> None:
    """Grouped bars: model, then horizon, then the q=0.05 / q=0.95 pair."""
    fig, ax = plt.subplots(figsize=(10.4, 4.85))
    bar_width = 0.34
    pair_gap = 0.06
    horizon_gap = 0.28
    model_gap = 1.05
    cursor = 0.0
    model_spans = []
    horizon_ticks = []
    for model_index, model in enumerate(INTERVAL_MODELS):
        bar_centers = []
        for horizon_index, horizon in enumerate(HORIZONS):
            pair_centers = []
            for quantile_index, quantile in enumerate(QUANTILES):
                score = value(data, f"d2_pinball_score__{horizon}__{quantile}", model)
                ax.bar(
                    cursor,
                    score,
                    width=bar_width,
                    color=QUANTILE_COLORS[quantile],
                    zorder=2,
                    edgecolor="white",
                    linewidth=0.4,
                    label=QUANTILE_LABELS[quantile] if model_index == 0 and horizon_index == 0 else "_nolegend_",
                )
                pair_centers.append(cursor)
                bar_centers.append(cursor)
                cursor += bar_width + (pair_gap if quantile_index == 0 else 0.0)
            horizon_ticks.append((sum(pair_centers) / len(pair_centers), horizon))
            if horizon_index < len(HORIZONS) - 1:
                cursor += horizon_gap
        model_spans.append((model, sum(bar_centers) / len(bar_centers)))
        if model_index < len(INTERVAL_MODELS) - 1:
            boundary = cursor + model_gap / 2
            ax.axvline(boundary, color="#e4ddd0", linewidth=0.8, zorder=1)
            cursor += model_gap

    ax.set_xticks([center for center, _label in horizon_ticks])
    ax.set_xticklabels([label for _center, label in horizon_ticks])
    ax.tick_params(axis="x", length=0, pad=3)
    for model, center in model_spans:
        ax.text(
            center,
            -0.14,
            model,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.6,
            color="#1c1915",
            clip_on=False,
        )
    ax.set_xlim(-0.28, cursor + 0.12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pinball D²")
    ax.set_title("Pinball D² of the 90% interval tails", loc="left")
    ax.yaxis.grid(True, color="#e4ddd0", linewidth=0.7)
    ax.xaxis.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=2, frameon=False, loc="upper right", borderaxespad=0.1)
    _save(fig, "d2_pinball_90")


def main() -> None:
    _style()
    data = load_metrics()
    plot_grouped_error(
        data,
        "mae",
        "MAE",
        "MAE by model and horizon",
        "mae_by_model",
        scale=1,
        formatter=_format_mae,
    )
    plot_grouped_error(
        data,
        "mape",
        "MAPE (%)",
        "MAPE by model and horizon",
        "mape_by_model",
        scale=100,
        formatter=lambda item: f"{item:.2f}",
    )
    for horizon in HORIZONS:
        plot_width_coverage(data, horizon)
    plot_d2(data)


if __name__ == "__main__":
    main()
