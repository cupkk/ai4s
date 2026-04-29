from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL


DEFAULT_PRICE_LAGS = [96, 192, 672]
DEFAULT_PRICE_ROLL_WINDOWS = [96, 672]


def lag_feature_names(lags: Sequence[int] = DEFAULT_PRICE_LAGS) -> List[str]:
    return [f"price_lag_{lag}" for lag in lags]


def roll_feature_names(windows: Sequence[int] = DEFAULT_PRICE_ROLL_WINDOWS) -> List[str]:
    names: List[str] = []
    for window in windows:
        names.extend([f"price_roll_{window}_mean", f"price_roll_{window}_std"])
    return names


def all_lag_feature_names(
    lags: Sequence[int] = DEFAULT_PRICE_LAGS,
    windows: Sequence[int] = DEFAULT_PRICE_ROLL_WINDOWS,
) -> List[str]:
    return lag_feature_names(lags) + roll_feature_names(windows)


def add_training_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: Sequence[int] = DEFAULT_PRICE_LAGS,
    windows: Sequence[int] = DEFAULT_PRICE_ROLL_WINDOWS,
) -> pd.DataFrame:
    out = df.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    out = out.sort_values(TIME_COL).reset_index(drop=True)

    for lag in lags:
        shifted = out[[TIME_COL, target_col]].copy()
        shifted[TIME_COL] = shifted[TIME_COL] + pd.Timedelta(minutes=15 * lag)
        shifted = shifted.rename(columns={target_col: f"price_lag_{lag}"})
        out = out.merge(shifted, on=TIME_COL, how="left")

    history = out[[TIME_COL, target_col]].sort_values(TIME_COL).copy()
    for window in windows:
        values = history[target_col].shift(1)
        history[f"price_roll_{window}_mean"] = values.rolling(window, min_periods=8).mean()
        history[f"price_roll_{window}_std"] = values.rolling(window, min_periods=8).std()
    roll_cols = [col for col in history.columns if col.startswith("price_roll_")]
    out = out.merge(history[[TIME_COL] + roll_cols], on=TIME_COL, how="left")
    return out


def recursive_lag_feature_frame(
    base_df: pd.DataFrame,
    history_prices: pd.DataFrame,
    predict_fn,
    feature_builder_fn,
    target_col: str = TARGET_COL,
    lags: Sequence[int] = DEFAULT_PRICE_LAGS,
    windows: Sequence[int] = DEFAULT_PRICE_ROLL_WINDOWS,
) -> np.ndarray:
    """Predict rows sequentially while filling lag features from actual/predicted history.

    `predict_fn(row_frame)` receives a one-row feature frame and returns one float.
    `feature_builder_fn(frame)` must return an aligned feature matrix for model prediction.
    """
    base = base_df.copy()
    base[TIME_COL] = pd.to_datetime(base[TIME_COL])
    base = base.sort_values(TIME_COL).reset_index(drop=True)

    hist = history_prices[[TIME_COL, target_col]].copy()
    hist[TIME_COL] = pd.to_datetime(hist[TIME_COL])
    price_by_time: Dict[pd.Timestamp, float] = {
        pd.Timestamp(t): float(v) for t, v in zip(hist[TIME_COL], hist[target_col])
    }

    ordered_prices = deque(
        [
            (pd.Timestamp(t), float(v))
            for t, v in sorted(price_by_time.items(), key=lambda item: item[0])
        ]
    )
    predictions: List[float] = []

    for _, row in base.iterrows():
        current_time = pd.Timestamp(row[TIME_COL])
        row_df = pd.DataFrame([row.to_dict()])
        for lag in lags:
            lag_time = current_time - pd.Timedelta(minutes=15 * lag)
            row_df[f"price_lag_{lag}"] = price_by_time.get(lag_time, np.nan)

        prior_values = [
            value for t, value in ordered_prices if t < current_time
        ]
        for window in windows:
            tail = np.asarray(prior_values[-window:], dtype=float)
            if tail.size:
                row_df[f"price_roll_{window}_mean"] = float(np.mean(tail))
                row_df[f"price_roll_{window}_std"] = float(np.std(tail))
            else:
                row_df[f"price_roll_{window}_mean"] = np.nan
                row_df[f"price_roll_{window}_std"] = np.nan

        pred = float(predict_fn(feature_builder_fn(row_df)))
        predictions.append(pred)
        price_by_time[current_time] = pred
        ordered_prices.append((current_time, pred))

    return np.asarray(predictions, dtype=float)

