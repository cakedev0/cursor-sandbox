"""Per-fold, per-horizon feature matrices of the upstream skrub pipeline.

The upstream ``make_data_op`` builds the features of each horizon with
``add_features`` and feeds ``ToFloat`` numpy arrays to the per-horizon
estimator. This module calls the same upstream functions directly so the
teacher predictions can be computed once and cached, and reused by many student
variants. ``check_folds.py`` checks that it reproduces the skrub pipeline.
"""

import datetime
import functools

import numpy as np
import polars as pl
import skrub

import electricity_load_forecasting as elf
from electricity_load_forecasting import pipeline as elp
from electricity_load_forecasting.splitter import TimeSeriesSplitter

HORIZONS = (1, 12, 24)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


@functools.lru_cache(maxsize=1)
def load_history():
    return elf.resample(elf.fetch_load_mw_history())


@functools.lru_cache(maxsize=1)
def prediction_times_and_targets():
    env = elf.get_env()
    prediction_time = elp.time_range(env["start"], env["end"])
    X_y = elp.get_X_y(prediction_time, load_history(), HORIZONS, mode="fit")
    return X_y["X"], X_y["y"]


@functools.lru_cache(maxsize=1)
def splits():
    X, _ = prediction_times_and_targets()
    return list(TimeSeriesSplitter().split(X))


@functools.lru_cache(maxsize=1)
def extended_splits():
    """Walk-forward splits with at least 1 year of training data.

    The test folds are aligned to the end of the data, so the last
    ``n_folds()`` of them are exactly the outer CV folds; the earlier ones
    (negative fold ids in ``fold_data``) only serve to calibrate the first
    outer folds on past out-of-fold predictions.
    """
    X, _ = prediction_times_and_targets()
    ext = list(TimeSeriesSplitter(min_train_days=365).split(X))
    n_extra = len(ext) - n_folds()
    for (a_train, a_test), (b_train, b_test) in zip(ext[n_extra:], splits()):
        assert (a_train == b_train).all() and (a_test == b_test).all()
    return {fold - n_extra: split for fold, split in enumerate(ext)}


def fold_ids(extended=True):
    return sorted(extended_splits()) if extended else list(range(n_folds()))


def _features(X, horizon):
    df = elp.add_features(
        X,
        horizon=horizon,
        temperature_only=True,
        city_names="all",
        load_mw_history=load_history(),
        city_weather_fetcher=elf.fetch_city_weather_forecast,
        holidays_fetcher=elp.fetch_holidays,
    ).drop(["prediction_time", "target_time"])
    return df


@functools.lru_cache(maxsize=8)
def all_features(horizon):
    """Features of all prediction times for one horizon (polars DataFrame)."""
    X, _ = prediction_times_and_targets()
    return _features(X, horizon)


def fold_data(fold, horizon):
    """Return a dict with train/test arrays for one CV fold and horizon.

    ``fold`` indexes the outer CV folds (0 to ``n_folds() - 1``); negative ids
    are the earlier calibration-only folds of ``extended_splits``.

    As in the skrub pipeline, the features of the train and test rows are
    computed separately on each subset and converted with ``skrub.ToFloat``
    (applied to all columns) fitted on the train rows.
    """
    X, y = prediction_times_and_targets()
    train, test = extended_splits()[fold]
    to_float = skrub.ApplyToCols(skrub.ToFloat())
    X_train_df = _features(X[train], horizon)
    X_test_df = _features(X[test], horizon)
    X_train = np.asarray(to_float.fit_transform(X_train_df), dtype=np.float64)
    X_test = np.asarray(to_float.transform(X_test_df), dtype=np.float64)
    return {
        "feature_names": X_train_df.columns,
        "X_train": X_train,
        "y_train": y[f"{horizon}h"].to_numpy()[train].astype(np.float64),
        "X_test": X_test,
        "y_test": y[f"{horizon}h"].to_numpy()[test].astype(np.float64),
        "time_train": X["prediction_time"].to_numpy()[train],
        "time_test": X["prediction_time"].to_numpy()[test],
        "train_idx": train,
        "test_idx": test,
    }


def last_observed_load(X, feature_names, horizon):
    """Load at prediction time (feature ``lag_<horizon>``), for residual targets.

    The few missing values fall back to the 24h rolling median of the same lag.
    """
    names = list(feature_names)
    base = X[:, names.index(f"lag_{horizon}")].copy()
    fallback = X[:, names.index(f"lag_{horizon}_width_24_median")]
    missing = np.isnan(base)
    base[missing] = fallback[missing]
    return np.nan_to_num(base, nan=float(np.nanmedian(base)))


def lag_difference_features(X, feature_names, horizon):
    """Append differences of load lags, which axis-aligned trees cannot form.

    Adds ``lag_k - lag_{k+1}`` for consecutive hourly lags (the recent slope
    and curvature of the load curve) and ``lag_h - lag_k`` for the daily and
    weekly lags (``k`` in 24, 48, 168), where ``lag_h`` is the load at
    prediction time.
    """
    names = list(feature_names)
    col = {n: X[:, names.index(n)] for n in names if n.startswith("lag_") and "width" not in n}
    hourly = sorted(int(n.removeprefix("lag_")) for n in col)
    new = [
        col[f"lag_{a}"] - col[f"lag_{b}"]
        for a, b in zip(hourly, hourly[1:])
        if b == a + 1
    ]
    new += [
        col[f"lag_{horizon}"] - col[f"lag_{k}"]
        for k in (24, 48, 168)
        if k != horizon and f"lag_{k}" in col
    ]
    return np.column_stack([X, *new]) if new else X


def n_folds():
    return len(splits())


def cutmix(X, ratio, seed):
    """CutMix synthetic queries: each feature is taken from one of two random rows.

    ``ratio`` synthetic rows are generated per real row. Each feature of a
    synthetic row comes, with probability 0.5, from one of two real rows drawn
    uniformly at random (arXiv:2610.01435, appendix A).
    """
    rng = np.random.default_rng(seed)
    n_syn = int(round(ratio * X.shape[0]))
    a = rng.integers(0, X.shape[0], size=n_syn)
    b = rng.integers(0, X.shape[0], size=n_syn)
    mask = rng.random((n_syn, X.shape[1])) < 0.5
    return np.where(mask, X[a], X[b])


def predictions_frame(fold_preds):
    """Assemble predictions in the upstream ``cross_validate_models.py`` format.

    ``fold_preds`` maps ``(fold, horizon)`` to an array of shape
    ``(n_test, n_quantiles)`` of sorted quantile predictions.
    """
    X, y = prediction_times_and_targets()
    frames = []
    for fold, (_, test) in enumerate(splits()):
        cols = {"prediction_time": X["prediction_time"].gather(test)}
        for h in HORIZONS:
            cols[f"{h}h"] = y[f"{h}h"].gather(test)
        for h in HORIZONS:
            pred = fold_preds[(fold, h)]
            for i, q in enumerate(QUANTILES):
                cols[f"pred_{h}h__q_{q}"] = pl.Series(pred[:, i], dtype=pl.Float64)
        frames.append(pl.DataFrame(cols).with_columns(split=pl.lit(fold)))
    return pl.concat(frames, how="vertical")


def utc(s):
    return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.UTC)
