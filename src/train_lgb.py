from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import (
    DEFAULT_BASE_FEATURES,
    TARGET_COL,
    TIME_COL,
    align_feature_frame,
    build_features,
    ensure_datetime,
    fit_history_stats,
)
from .nwp_features import load_or_build_nwp_features, merge_nwp_features
from .validate_profit import parse_threshold_grid, search_best_threshold


DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "min_data_in_leaf": 32,
    "seed": 42,
    "verbose": -1,
}


def load_training_frame(
    train_feature_path: str,
    train_label_path: str,
    target_col: str = TARGET_COL,
    nwp_dir: str = "",
    nwp_cache: str = "outputs/nwp_features.csv",
) -> pd.DataFrame:
    feat = pd.read_csv(train_feature_path)
    label = pd.read_csv(train_label_path)
    if TIME_COL not in feat.columns or TIME_COL not in label.columns:
        raise ValueError("both feature and label CSV files must contain a times column")
    if target_col not in label.columns:
        raise ValueError(f"label CSV does not contain target column: {target_col}")

    feat[TIME_COL] = pd.to_datetime(feat[TIME_COL])
    label[TIME_COL] = pd.to_datetime(label[TIME_COL])

    if nwp_dir:
        all_times = pd.concat([feat[[TIME_COL]], label[[TIME_COL]]], ignore_index=True)
        nwp = load_or_build_nwp_features(
            nwp_dir,
            nwp_cache,
            start_time=str(all_times[TIME_COL].min()),
            end_time=str(all_times[TIME_COL].max()),
        )
        feat = merge_nwp_features(feat, nwp)

    df = pd.merge(feat, label[[TIME_COL, target_col]], on=TIME_COL, how="inner")
    df = ensure_datetime(df).sort_values(TIME_COL).reset_index(drop=True)
    if df.empty:
        raise ValueError("merged training data is empty")
    return df


def parse_seeds(text: str) -> List[int]:
    seeds: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _parse_date(value: str) -> Optional[pd.Timestamp]:
    if not value:
        return None
    return pd.Timestamp(value)


def split_by_day(
    df: pd.DataFrame,
    val_ratio: float,
    val_start_date: str = "",
    val_end_date: str = "",
    val_days: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df[TIME_COL].dt.date.unique())
    if len(dates) < 2:
        raise ValueError("need at least two days for day-based validation")

    start = _parse_date(val_start_date)
    end = _parse_date(val_end_date)
    if start is not None or end is not None:
        start_date = (start or pd.Timestamp.min).date()
        end_date = (end or pd.Timestamp.max).date()
        val_dates = {d for d in dates if start_date <= d <= end_date}
    else:
        n_val = val_days if val_days > 0 else max(1, int(round(len(dates) * val_ratio)))
        n_val = min(n_val, len(dates) - 1)
        val_dates = set(dates[-n_val:])

    if not val_dates:
        raise ValueError("validation date selection is empty")
    if len(val_dates) >= len(dates):
        raise ValueError("validation date selection leaves no training days")

    train = df[~df[TIME_COL].dt.date.isin(val_dates)].copy()
    val = df[df[TIME_COL].dt.date.isin(val_dates)].copy()
    return train, val


def params_for_seed(params: Dict[str, object], seed: int) -> Dict[str, object]:
    out = params.copy()
    out["seed"] = seed
    out["feature_fraction_seed"] = seed
    out["bagging_seed"] = seed
    out["data_random_seed"] = seed
    out["drop_seed"] = seed
    return out


def train_booster(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame | None,
    val_y: np.ndarray | None,
    params: Dict[str, object],
    num_boost_round: int,
    early_stopping_rounds: int,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed; run: pip install -r requirements.txt") from exc

    train_set = lgb.Dataset(train_x, label=train_y, feature_name=list(train_x.columns))
    valid_sets = None
    callbacks = [lgb.log_evaluation(100)]
    if val_x is not None and val_y is not None:
        val_set = lgb.Dataset(val_x, label=val_y, feature_name=list(train_x.columns), reference=train_set)
        valid_sets = [val_set]
        callbacks.insert(0, lgb.early_stopping(early_stopping_rounds))

    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def save_metadata(path: str, metadata: Dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def save_feature_importance(path: str, feature_columns: List[str], importances: List[np.ndarray]) -> None:
    if not importances:
        return
    arr = np.vstack(importances)
    out = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": arr.mean(axis=0),
            "importance_std": arr.std(axis=0),
        }
    ).sort_values("importance_mean", ascending=False)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def model_path_for_seed(model_output: str, seed: int, ensemble_size: int) -> str:
    if ensemble_size == 1:
        return model_output
    path = Path(model_output)
    return str(path.with_name(f"{path.stem}_seed{seed}{path.suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM price model and tune profit threshold.")
    parser.add_argument("--train-feature", required=True, help="Training boundary condition CSV.")
    parser.add_argument("--train-label", required=True, help="Training node price CSV.")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--nwp-dir", default="", help="Optional official all_nc directory.")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features.csv")
    parser.add_argument("--model-output", default="outputs/lgb_model.txt")
    parser.add_argument("--metadata-output", default="outputs/lgb_model_metadata.json")
    parser.add_argument("--val-pred-output", default="outputs/val_predictions.csv")
    parser.add_argument("--threshold-output", default="outputs/best_threshold.txt")
    parser.add_argument("--feature-importance-output", default="outputs/feature_importance.csv")
    parser.add_argument("--threshold-grid", default="0,5000,10000,20000,30000,50000,80000,100000")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-days", type=int, default=0, help="Use the last N days for validation.")
    parser.add_argument("--val-start-date", default="", help="Inclusive validation start date, e.g. 2025-01-01.")
    parser.add_argument("--val-end-date", default="", help="Inclusive validation end date, e.g. 2025-02-28.")
    parser.add_argument("--seeds", default="42,2024,2026", help="Comma-separated LightGBM ensemble seeds.")
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--use-exact-calendar-history", action="store_true")
    parser.add_argument("--use-forecast-bias", action="store_true")
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--params-json", default="", help="Optional JSON object overriding LightGBM params.")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    if args.params_json:
        params.update(json.loads(args.params_json))

    seeds = parse_seeds(args.seeds)
    df = load_training_frame(
        args.train_feature,
        args.train_label,
        target_col=args.target_col,
        nwp_dir=args.nwp_dir,
        nwp_cache=args.nwp_cache,
    )
    train_df, val_df = split_by_day(
        df,
        args.val_ratio,
        val_start_date=args.val_start_date,
        val_end_date=args.val_end_date,
        val_days=args.val_days,
    )
    print(
        "validation_period="
        f"{val_df[TIME_COL].min()}..{val_df[TIME_COL].max()}, "
        f"rows={len(val_df)}, days={val_df[TIME_COL].dt.date.nunique()}"
    )

    train_stats = fit_history_stats(train_df, target_col=args.target_col)
    train_features = build_features(
        train_df,
        history_stats=train_stats,
        use_exact_calendar_history=args.use_exact_calendar_history,
        use_forecast_bias=args.use_forecast_bias,
    )
    val_features = build_features(
        val_df,
        history_stats=train_stats,
        use_exact_calendar_history=args.use_exact_calendar_history,
        use_forecast_bias=args.use_forecast_bias,
    )
    feature_columns = train_features.feature_columns

    train_x = align_feature_frame(train_features.frame, feature_columns)
    val_x = align_feature_frame(val_features.frame, feature_columns)
    train_y = train_df[args.target_col].to_numpy(dtype=float)
    val_y = val_df[args.target_col].to_numpy(dtype=float)

    val_preds: List[np.ndarray] = []
    best_iterations: List[int] = []
    for seed in seeds:
        print(f"training_validation_model_seed={seed}")
        val_model = train_booster(
            train_x,
            train_y,
            val_x,
            val_y,
            params=params_for_seed(params, seed),
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        best_iteration = int(val_model.best_iteration or args.num_boost_round)
        best_iterations.append(best_iteration)
        val_preds.append(val_model.predict(val_x, num_iteration=best_iteration))

    val_pred = np.mean(np.vstack(val_preds), axis=0)

    val_pred_data = {
        TIME_COL: val_df[TIME_COL].to_numpy(),
        args.target_col: val_y,
        "pred_price": val_pred,
    }
    for seed, pred in zip(seeds, val_preds):
        val_pred_data[f"pred_price_seed{seed}"] = pred
    val_pred_df = pd.DataFrame(val_pred_data)
    Path(args.val_pred_output).parent.mkdir(parents=True, exist_ok=True)
    val_pred_df.to_csv(args.val_pred_output, index=False)

    thresholds = parse_threshold_grid(args.threshold_grid)
    best_threshold, threshold_summary = search_best_threshold(
        val_pred_df,
        thresholds,
        pred_col="pred_price",
        true_col=args.target_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
    )
    threshold_summary_path = Path(args.val_pred_output).with_name("threshold_search.csv")
    threshold_summary.to_csv(threshold_summary_path, index=False)

    print(f"validation_rmse={rmse(val_y, val_pred):.6f}")
    print(f"validation_mae={mae(val_y, val_pred):.6f}")
    print(threshold_summary.to_string(index=False))
    print(
        "best_threshold="
        f"{best_threshold['threshold']}, avg_profit={best_threshold['avg_profit']:.6f}"
    )

    full_stats = fit_history_stats(df, target_col=args.target_col)
    full_features = build_features(
        df,
        history_stats=full_stats,
        use_exact_calendar_history=args.use_exact_calendar_history,
        use_forecast_bias=args.use_forecast_bias,
    )
    full_feature_columns = full_features.feature_columns
    full_x = align_feature_frame(full_features.frame, full_feature_columns)
    full_y = df[args.target_col].to_numpy(dtype=float)

    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    model_paths: List[str] = []
    final_importances: List[np.ndarray] = []
    for seed, final_rounds in zip(seeds, best_iterations):
        print(f"training_final_model_seed={seed}, rounds={final_rounds}")
        final_model = train_booster(
            full_x,
            full_y,
            None,
            None,
            params=params_for_seed(params, seed),
            num_boost_round=final_rounds,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        model_path = model_path_for_seed(args.model_output, seed, len(seeds))
        final_model.save_model(model_path)
        model_paths.append(model_path)
        final_importances.append(final_model.feature_importance(importance_type="gain"))

    save_feature_importance(args.feature_importance_output, full_feature_columns, final_importances)

    save_metadata(
        args.metadata_output,
        {
            "target_col": args.target_col,
            "feature_columns": full_feature_columns,
            "base_features": DEFAULT_BASE_FEATURES,
            "history_stats": full_stats,
            "model_paths": model_paths,
            "seeds": seeds,
            "best_threshold": float(best_threshold["threshold"]),
            "strategy_constraints": {
                "charge_start_min": args.charge_start_min,
                "charge_start_max": args.charge_start_max,
                "discharge_start_min": args.discharge_start_min,
                "discharge_start_max": args.discharge_start_max,
            },
            "feature_options": {
                "use_exact_calendar_history": args.use_exact_calendar_history,
                "use_forecast_bias": args.use_forecast_bias,
                "use_nwp": bool(args.nwp_dir),
            },
            "validation": {
                "rmse": rmse(val_y, val_pred),
                "mae": mae(val_y, val_pred),
                "avg_profit": float(best_threshold["avg_profit"]),
                "avg_oracle_profit": float(best_threshold.get("avg_oracle_profit", 0.0)),
                "capture_ratio": float(best_threshold.get("capture_ratio", 0.0)),
                "avg_regret": float(best_threshold.get("avg_regret", 0.0)),
                "window_hit_rate": float(best_threshold.get("window_hit_rate", 0.0)),
                "window_hit_2_rate": float(best_threshold.get("window_hit_2_rate", 0.0)),
                "traded_days": int(best_threshold["traded_days"]),
                "oracle_traded_days": int(best_threshold.get("oracle_traded_days", 0)),
                "loss_days": int(best_threshold["loss_days"]),
                "skipped_incomplete_days": int(best_threshold["skipped_incomplete_days"]),
                "period_start": str(val_df[TIME_COL].min()),
                "period_end": str(val_df[TIME_COL].max()),
            },
            "lightgbm_params": params,
            "num_boost_rounds": best_iterations,
        },
    )
    Path(args.threshold_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.threshold_output).write_text(str(float(best_threshold["threshold"])), encoding="utf-8")
    print(f"saved_models={model_paths}")
    print(f"saved_metadata={args.metadata_output}")


if __name__ == "__main__":
    main()
