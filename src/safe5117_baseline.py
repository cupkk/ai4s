from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, align_feature_frame, build_features, fit_history_stats
from .price_history_features import add_price_history_features, fit_price_history_features
from .storage_optimizer import generate_strategy
from .train_lgb import DEFAULT_PARAMS, params_for_seed, train_booster


SOURCE_PRED_COL = "safe5117_source_pred_price"


@dataclass
class Safe5117SourceBaseline:
    train_meta: pd.DataFrame
    predict_meta: pd.DataFrame
    train_predictions: pd.DataFrame
    predict_predictions: pd.DataFrame
    diagnostics: dict[str, Any]


def baseline_meta_from_price_predictions(
    prediction_df: pd.DataFrame,
    price_col: str = SOURCE_PRED_COL,
    threshold: float = -1.0e18,
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
) -> pd.DataFrame:
    if TIME_COL not in prediction_df.columns:
        raise ValueError(f"prediction frame missing {TIME_COL}")
    if price_col not in prediction_df.columns:
        raise ValueError(f"prediction frame missing price column: {price_col}")

    pred = prediction_df[[TIME_COL, price_col]].copy()
    pred[TIME_COL] = pd.to_datetime(pred[TIME_COL])
    pred["__date__"] = pred[TIME_COL].dt.date
    day_counts = pred.groupby("__date__", sort=True).size()
    complete_dates = set(day_counts[day_counts == 96].index)
    if not complete_dates:
        raise ValueError("no complete 96-point days available for safe5117 source baseline meta")

    pred = pred.loc[pred["__date__"].isin(complete_dates), [TIME_COL, price_col]].copy()
    _, meta = generate_strategy(
        pred,
        threshold=threshold,
        price_col=price_col,
        charge_start_min=charge_start_min,
        charge_start_max=charge_start_max,
        discharge_start_min=discharge_start_min,
        discharge_start_max=discharge_start_max,
    )
    meta = meta.rename(
        columns={
            "charge_start": "baseline_charge_start",
            "discharge_start": "baseline_discharge_start",
        }
    )
    return meta.sort_values("date").reset_index(drop=True)


def _split_source_validation(
    df: pd.DataFrame,
    val_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(df[TIME_COL]).dt.date.unique())
    if len(dates) < 3:
        raise ValueError("safe5117 source model needs at least 3 training days")
    n_val = max(1, min(int(val_days), len(dates) - 1))
    val_dates = set(dates[-n_val:])
    train = df.loc[~pd.to_datetime(df[TIME_COL]).dt.date.isin(val_dates)].copy()
    val = df.loc[pd.to_datetime(df[TIME_COL]).dt.date.isin(val_dates)].copy()
    return train, val


def _feature_matrix(
    frame: pd.DataFrame,
    price_stats: dict[str, object],
    history_stats: dict[str, object],
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    model_df = add_price_history_features(frame, price_stats)
    features = build_features(model_df, history_stats=history_stats)
    columns = list(feature_columns) if feature_columns is not None else features.feature_columns
    return align_feature_frame(features.frame, columns), columns


def fit_safe5117_source_baseline(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    seeds: Sequence[int] = (42,),
    params: dict[str, object] | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    source_val_days: int = 59,
    train_baseline_mode: str = "recent-oof",
    threshold: float = 0.0,
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
) -> Safe5117SourceBaseline:
    if train_df.empty or predict_df.empty:
        raise ValueError("safe5117 source baseline requires non-empty train and predict frames")
    if target_col not in train_df.columns:
        raise ValueError(f"safe5117 source baseline training frame missing target column: {target_col}")

    work_train = train_df.copy()
    work_predict = predict_df.copy()
    work_train[TIME_COL] = pd.to_datetime(work_train[TIME_COL])
    work_predict[TIME_COL] = pd.to_datetime(work_predict[TIME_COL])
    source_train, source_val = _split_source_validation(work_train, source_val_days)

    base_params = DEFAULT_PARAMS.copy()
    if params:
        base_params.update(params)

    price_stats = fit_price_history_features(source_train, target_col=target_col)
    source_train_model = add_price_history_features(source_train, price_stats)
    source_history_stats = fit_history_stats(source_train_model, target_col=target_col)
    source_train_x, source_feature_columns = _feature_matrix(
        source_train,
        price_stats,
        source_history_stats,
    )
    source_val_x, _ = _feature_matrix(
        source_val,
        price_stats,
        source_history_stats,
        source_feature_columns,
    )
    source_train_y = source_train[target_col].to_numpy(dtype=float)
    source_val_y = source_val[target_col].to_numpy(dtype=float)

    best_iterations: list[int] = []
    source_val_pred_parts: list[np.ndarray] = []
    for seed in seeds:
        model = train_booster(
            source_train_x,
            source_train_y,
            source_val_x,
            source_val_y,
            params=params_for_seed(base_params, int(seed)),
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        best_iteration = int(model.best_iteration or num_boost_round)
        best_iterations.append(best_iteration)
        source_val_pred_parts.append(model.predict(source_val_x, num_iteration=best_iteration))

    full_price_stats = fit_price_history_features(work_train, target_col=target_col)
    full_train_model = add_price_history_features(work_train, full_price_stats)
    full_history_stats = fit_history_stats(full_train_model, target_col=target_col)
    full_train_x, full_feature_columns = _feature_matrix(
        work_train,
        full_price_stats,
        full_history_stats,
    )
    predict_x, _ = _feature_matrix(
        work_predict,
        full_price_stats,
        full_history_stats,
        full_feature_columns,
    )
    full_train_y = work_train[target_col].to_numpy(dtype=float)

    train_pred_parts: list[np.ndarray] = []
    predict_pred_parts: list[np.ndarray] = []
    for seed, rounds in zip(seeds, best_iterations):
        model = train_booster(
            full_train_x,
            full_train_y,
            None,
            None,
            params=params_for_seed(base_params, int(seed)),
            num_boost_round=int(rounds),
            early_stopping_rounds=early_stopping_rounds,
        )
        train_pred_parts.append(model.predict(full_train_x, num_iteration=int(rounds)))
        predict_pred_parts.append(model.predict(predict_x, num_iteration=int(rounds)))

    train_pred = np.mean(np.vstack(train_pred_parts), axis=0)
    predict_pred = np.mean(np.vstack(predict_pred_parts), axis=0)
    if train_baseline_mode == "recent-oof":
        train_times = source_val[TIME_COL].to_numpy()
        train_meta_pred = np.mean(np.vstack(source_val_pred_parts), axis=0)
        train_meta_std = np.std(np.vstack(source_val_pred_parts), axis=0)
    elif train_baseline_mode == "in-sample":
        train_times = work_train[TIME_COL].to_numpy()
        train_meta_pred = train_pred
        train_meta_std = np.std(np.vstack(train_pred_parts), axis=0)
    else:
        raise ValueError(f"unsupported train_baseline_mode: {train_baseline_mode}")

    train_predictions = pd.DataFrame({TIME_COL: train_times, SOURCE_PRED_COL: train_meta_pred})
    predict_predictions = pd.DataFrame(
        {
            TIME_COL: work_predict[TIME_COL].to_numpy(),
            SOURCE_PRED_COL: predict_pred,
        }
    )
    if len(train_pred_parts) > 1:
        train_predictions[f"{SOURCE_PRED_COL}_std"] = train_meta_std
        predict_predictions[f"{SOURCE_PRED_COL}_std"] = np.std(np.vstack(predict_pred_parts), axis=0)

    train_meta = baseline_meta_from_price_predictions(
        train_predictions,
        price_col=SOURCE_PRED_COL,
        threshold=threshold,
        charge_start_min=charge_start_min,
        charge_start_max=charge_start_max,
        discharge_start_min=discharge_start_min,
        discharge_start_max=discharge_start_max,
    )
    predict_meta = baseline_meta_from_price_predictions(
        predict_predictions,
        price_col=SOURCE_PRED_COL,
        threshold=threshold,
        charge_start_min=charge_start_min,
        charge_start_max=charge_start_max,
        discharge_start_min=discharge_start_min,
        discharge_start_max=discharge_start_max,
    )
    diagnostics = {
        "mode": "safe5117-source-model",
        "seeds": [int(seed) for seed in seeds],
        "best_iterations": best_iterations,
        "source_val_days": int(source_val[TIME_COL].dt.date.nunique()),
        "train_baseline_mode": train_baseline_mode,
        "train_days": int(work_train[TIME_COL].dt.date.nunique()),
        "predict_days": int(work_predict[TIME_COL].dt.date.nunique()),
        "train_meta_days": int(len(train_meta)),
        "predict_meta_days": int(len(predict_meta)),
        "threshold": float(threshold),
        "strategy_constraints": {
            "charge_start_min": int(charge_start_min),
            "charge_start_max": int(charge_start_max),
            "discharge_start_min": int(discharge_start_min),
            "discharge_start_max": int(discharge_start_max),
        },
    }
    return Safe5117SourceBaseline(
        train_meta=train_meta,
        predict_meta=predict_meta,
        train_predictions=train_predictions,
        predict_predictions=predict_predictions,
        diagnostics=diagnostics,
    )
