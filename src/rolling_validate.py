from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, align_feature_frame, build_features, fit_history_stats
from .train_lgb import (
    DEFAULT_PARAMS,
    load_training_frame,
    mae,
    params_for_seed,
    parse_seeds,
    rmse,
    train_booster,
)
from .validate_profit import backtest_predictions


DEFAULT_FOLDS = "2025-04-01:2025-04-30,2025-07-01:2025-07-31,2025-10-01:2025-10-31,2025-12-01:2025-12-31"
DEFAULT_STRATEGIES = [
    {
        "strategy": "unconstrained",
        "threshold": 0.0,
        "charge_start_min": 0,
        "charge_start_max": 80,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
    },
    {
        "strategy": "c0_55",
        "threshold": 0.0,
        "charge_start_min": 0,
        "charge_start_max": 55,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
    },
    {
        "strategy": "c0_55_d72_88",
        "threshold": 0.0,
        "charge_start_min": 0,
        "charge_start_max": 55,
        "discharge_start_min": 72,
        "discharge_start_max": 88,
    },
    {
        "strategy": "threshold_1000",
        "threshold": 1000.0,
        "charge_start_min": 0,
        "charge_start_max": 80,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
    },
]


def parse_folds(text: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    folds: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        start, end = [part.strip() for part in item.split(":", 1)]
        folds.append((pd.Timestamp(start), pd.Timestamp(end)))
    if not folds:
        raise ValueError("at least one fold is required")
    return folds


def parse_strategy(text: str) -> Dict[str, object]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 6:
        raise ValueError(
            "--strategy format: name,threshold,charge_min,charge_max,discharge_min,discharge_max"
        )
    return {
        "strategy": parts[0],
        "threshold": float(parts[1]),
        "charge_start_min": int(parts[2]),
        "charge_start_max": int(parts[3]),
        "discharge_start_min": int(parts[4]),
        "discharge_start_max": int(parts[5]),
    }


def aggregate_results(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    numeric_cols = [
        "avg_profit",
        "capture_ratio",
        "avg_regret",
        "window_hit_rate",
        "window_hit_2_rate",
        "rmse",
        "mae",
        "loss_days",
    ]
    rows = []
    for strategy, group in detail.groupby("strategy", sort=True):
        row = {"strategy": strategy, "folds": int(len(group))}
        for col in numeric_cols:
            if col in group.columns:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_min"] = float(group[col].min())
                row[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows)
    if "avg_profit_mean" in out.columns:
        out = out.sort_values(["avg_profit_mean", "capture_ratio_mean"], ascending=[False, False])
    return out


def run_rolling_validation(
    df: pd.DataFrame,
    folds: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
    strategies: Sequence[Dict[str, object]],
    seeds: Sequence[int],
    params: Dict[str, object],
    num_boost_round: int,
    early_stopping_rounds: int,
    use_exact_calendar_history: bool,
    use_forecast_bias: bool,
    pred_output_dir: str,
) -> pd.DataFrame:
    rows = []
    pred_dir = Path(pred_output_dir) if pred_output_dir else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for fold_index, (start, end) in enumerate(folds, start=1):
        train_df = df[df[TIME_COL] < start].copy()
        val_df = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end + pd.Timedelta(hours=23, minutes=45))].copy()
        if train_df.empty or val_df.empty:
            raise ValueError(f"fold {fold_index} has empty train or validation data: {start}..{end}")

        stats = fit_history_stats(train_df, target_col=TARGET_COL)
        train_features = build_features(
            train_df,
            history_stats=stats,
            use_exact_calendar_history=use_exact_calendar_history,
            use_forecast_bias=use_forecast_bias,
        )
        val_features = build_features(
            val_df,
            history_stats=stats,
            use_exact_calendar_history=use_exact_calendar_history,
            use_forecast_bias=use_forecast_bias,
        )
        feature_columns = train_features.feature_columns
        train_x = align_feature_frame(train_features.frame, feature_columns)
        val_x = align_feature_frame(val_features.frame, feature_columns)
        train_y = train_df[TARGET_COL].to_numpy(dtype=float)
        val_y = val_df[TARGET_COL].to_numpy(dtype=float)

        pred_parts: List[np.ndarray] = []
        for seed in seeds:
            model = train_booster(
                train_x,
                train_y,
                val_x,
                val_y,
                params=params_for_seed(params, seed),
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
            best_iteration = int(model.best_iteration or num_boost_round)
            pred_parts.append(model.predict(val_x, num_iteration=best_iteration))

        pred = np.mean(np.vstack(pred_parts), axis=0)
        pred_df = pd.DataFrame(
            {TIME_COL: val_df[TIME_COL].to_numpy(), TARGET_COL: val_y, "pred_price": pred}
        )
        if pred_dir:
            pred_df.to_csv(pred_dir / f"fold_{fold_index:02d}_predictions.csv", index=False)

        fold_rmse = rmse(val_y, pred)
        fold_mae = mae(val_y, pred)
        for strategy in strategies:
            summary, _ = backtest_predictions(
                pred_df,
                threshold=float(strategy["threshold"]),
                pred_col="pred_price",
                true_col=TARGET_COL,
                charge_start_min=int(strategy["charge_start_min"]),
                charge_start_max=int(strategy["charge_start_max"]),
                discharge_start_min=int(strategy["discharge_start_min"]),
                discharge_start_max=int(strategy["discharge_start_max"]),
            )
            rows.append(
                {
                    "fold": fold_index,
                    "val_start": str(start.date()),
                    "val_end": str(end.date()),
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "rmse": fold_rmse,
                    "mae": fold_mae,
                    **strategy,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanding-window validation for price and strategy models.")
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--nwp-dir", default="")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features_train.csv")
    parser.add_argument("--folds", default=DEFAULT_FOLDS)
    parser.add_argument("--strategy", action="append", default=[])
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--use-exact-calendar-history", action="store_true")
    parser.add_argument("--use-forecast-bias", action="store_true")
    parser.add_argument("--params-json", default="")
    parser.add_argument("--output", default="outputs/rolling_validation.csv")
    parser.add_argument("--aggregate-output", default="outputs/rolling_validation_summary.csv")
    parser.add_argument("--pred-output-dir", default="")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    if args.params_json:
        params.update(json.loads(args.params_json))

    df = load_training_frame(
        args.train_feature,
        args.train_label,
        target_col=TARGET_COL,
        nwp_dir=args.nwp_dir,
        nwp_cache=args.nwp_cache,
    )
    folds = parse_folds(args.folds)
    strategies = [parse_strategy(item) for item in args.strategy] if args.strategy else DEFAULT_STRATEGIES
    detail = run_rolling_validation(
        df=df,
        folds=folds,
        strategies=strategies,
        seeds=parse_seeds(args.seeds),
        params=params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        use_exact_calendar_history=args.use_exact_calendar_history,
        use_forecast_bias=args.use_forecast_bias,
        pred_output_dir=args.pred_output_dir,
    )
    aggregate = aggregate_results(detail)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_path, index=False)
    aggregate_path = Path(args.aggregate_output)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(aggregate_path, index=False)
    print(aggregate.to_string(index=False))
    print(f"saved_rolling_detail={output_path}")
    print(f"saved_rolling_summary={aggregate_path}")


if __name__ == "__main__":
    main()
