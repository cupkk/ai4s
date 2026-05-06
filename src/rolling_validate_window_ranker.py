from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, build_features, fit_history_stats
from .price_history_features import add_price_history_features, fit_price_history_features
from .train_lgb import DEFAULT_PARAMS, load_training_frame, params_for_seed, parse_seeds
from .train_window_ranker import (
    DEFAULT_POINT_FEATURES,
    DEFAULT_BASELINE_WINDOW_SCORE_COL,
    _add_prediction_columns,
    _add_rank_columns,
    _group_sizes_by_date,
    _ranker_labels,
    _train_window_booster,
    _validation_day_metrics,
    attach_baseline_window_context,
    build_window_dataset,
    prepare_baseline_windows,
)


DEFAULT_FOLDS = "2025-04-01:2025-04-30,2025-07-01:2025-07-31,2025-10-01:2025-10-31,2025-12-01:2025-12-31"


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


def aggregate_fold_metrics(day_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, group in day_metrics.groupby("fold", sort=True):
        oracle_sum = float(group["oracle_profit"].sum())
        selected_sum = float(group["selected_profit"].sum())
        rows.append(
            {
                "fold": int(fold),
                "val_start": group["val_start"].iloc[0],
                "val_end": group["val_end"].iloc[0],
                "days": int(len(group)),
                "avg_profit": float(group["selected_profit"].mean()),
                "avg_delta_profit": float(group["selected_delta_profit"].mean())
                if "selected_delta_profit" in group.columns
                else np.nan,
                "positive_delta_rate": float((group["selected_delta_profit"] > 0).mean())
                if "selected_delta_profit" in group.columns
                else np.nan,
                "capture_ratio": selected_sum / oracle_sum if oracle_sum else 0.0,
                "regret": float(group["regret"].mean()),
                "delta_regret": float(group["delta_regret"].mean())
                if "delta_regret" in group.columns
                else np.nan,
                "top1_window_hit": float(group["top1_window_hit"].mean()),
                "top3_window_hit": float(group["top3_window_hit"].mean()),
                "top1_minus_top2_margin": float(group["top1_minus_top2_margin"].mean()),
                "negative_selected_days": int((group["selected_profit"] < 0).sum()),
                "p25_selected_profit": float(group["selected_profit"].quantile(0.25)),
                "p50_selected_profit": float(group["selected_profit"].quantile(0.50)),
                "p75_selected_profit": float(group["selected_profit"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def run_window_ranker_rolling_validation(
    df: pd.DataFrame,
    folds: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
    seeds: Sequence[int],
    params: Dict[str, object],
    objective_mode: str,
    label_mode: str,
    baseline_window_score_col: str,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    ranked_output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_day_metrics = []
    ranked_dir = Path(ranked_output_dir) if ranked_output_dir else None
    if ranked_dir:
        ranked_dir.mkdir(parents=True, exist_ok=True)

    for fold_index, (start, end) in enumerate(folds, start=1):
        train_df = df[df[TIME_COL] < start].copy()
        val_df = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end + pd.Timedelta(hours=23, minutes=45))].copy()
        if train_df.empty or val_df.empty:
            raise ValueError(f"fold {fold_index} has empty train or validation data: {start}..{end}")

        price_stats = fit_price_history_features(train_df, target_col=TARGET_COL)
        train_model_df = add_price_history_features(train_df, price_stats)
        val_model_df = add_price_history_features(val_df, price_stats)
        hist_stats = fit_history_stats(train_model_df, target_col=TARGET_COL)
        train_features = build_features(train_model_df, history_stats=hist_stats)
        val_features = build_features(val_model_df, history_stats=hist_stats)
        point_features = [col for col in DEFAULT_POINT_FEATURES if col in train_features.frame.columns]
        if not point_features:
            raise ValueError(f"fold {fold_index}: no point features available")

        train_windows, _ = build_window_dataset(
            train_df,
            train_features.frame,
            point_features,
            TARGET_COL,
            charge_start_min,
            charge_start_max,
            discharge_start_min,
            discharge_start_max,
            include_target=True,
        )
        val_windows, _ = build_window_dataset(
            val_df,
            val_features.frame,
            point_features,
            TARGET_COL,
            charge_start_min,
            charge_start_max,
            discharge_start_min,
            discharge_start_max,
            include_target=True,
        )
        if label_mode == "baseline-delta":
            train_baseline = prepare_baseline_windows(
                train_windows,
                score_col=baseline_window_score_col,
            )
            val_baseline = prepare_baseline_windows(
                val_windows,
                score_col=baseline_window_score_col,
            )
            train_windows = attach_baseline_window_context(train_windows, train_baseline)
            val_windows = attach_baseline_window_context(val_windows, val_baseline)
        label_col = "true_delta_profit" if label_mode == "baseline-delta" else "true_window_profit"
        feature_columns = [
            col
            for col in train_windows.columns
            if col not in {"date", "true_window_profit", "true_delta_profit", "baseline_true_window_profit"}
        ]
        train_x = train_windows[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        val_x = val_windows[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        train_y = train_windows[label_col].to_numpy(dtype=float)
        val_y = val_windows[label_col].to_numpy(dtype=float)
        train_group = _group_sizes_by_date(train_windows)
        val_group = _group_sizes_by_date(val_windows)
        train_rank_labels = _ranker_labels(train_windows, target_col=label_col)
        val_rank_labels = _ranker_labels(val_windows, target_col=label_col)

        val_preds = []
        for seed in seeds:
            model = _train_window_booster(
                train_x,
                train_y,
                val_x,
                val_y,
                params=params_for_seed(params, seed),
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                objective_mode=objective_mode,
                train_group=train_group,
                val_group=val_group,
                train_rank_labels=train_rank_labels,
                val_rank_labels=val_rank_labels,
            )
            best_iteration = int(model.best_iteration or num_boost_round)
            val_preds.append(model.predict(val_x, num_iteration=best_iteration))

        ranked = _add_prediction_columns(val_windows, val_preds, seeds)
        ranked = _add_rank_columns(ranked)
        if ranked_dir:
            ranked.to_csv(ranked_dir / f"fold_{fold_index:02d}_ranked_windows.csv", index=False)

        day_metrics = _validation_day_metrics(ranked, label_col=label_col)
        day_metrics.insert(0, "fold", fold_index)
        day_metrics.insert(1, "val_start", str(start.date()))
        day_metrics.insert(2, "val_end", str(end.date()))
        all_day_metrics.append(day_metrics)

    detail = pd.concat(all_day_metrics, ignore_index=True)
    return detail, aggregate_fold_metrics(detail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling validation for the direct window-profit model.")
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--nwp-dir", default="")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features_train.csv")
    parser.add_argument("--folds", default=DEFAULT_FOLDS)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--objective-mode", choices=["regression", "lambdarank"], default="regression")
    parser.add_argument("--label-mode", choices=["absolute", "baseline-delta"], default="absolute")
    parser.add_argument("--baseline-window-score-col", default=DEFAULT_BASELINE_WINDOW_SCORE_COL)
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--params-json", default="")
    parser.add_argument("--output", default="outputs/window_ranker_rolling_day_metrics.csv")
    parser.add_argument("--aggregate-output", default="outputs/window_ranker_rolling_summary.csv")
    parser.add_argument("--ranked-output-dir", default="")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    if args.objective_mode == "regression":
        params["objective"] = "regression"
        params["metric"] = "rmse"
    if args.params_json:
        params.update(json.loads(args.params_json))

    df = load_training_frame(
        args.train_feature,
        args.train_label,
        target_col=TARGET_COL,
        nwp_dir=args.nwp_dir,
        nwp_cache=args.nwp_cache,
    )
    detail, aggregate = run_window_ranker_rolling_validation(
        df=df,
        folds=parse_folds(args.folds),
        seeds=parse_seeds(args.seeds),
        params=params,
        objective_mode=args.objective_mode,
        label_mode=args.label_mode,
        baseline_window_score_col=args.baseline_window_score_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        ranked_output_dir=args.ranked_output_dir,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_path, index=False)
    aggregate_path = Path(args.aggregate_output)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(aggregate_path, index=False)
    print(aggregate.to_string(index=False))
    print(f"saved_window_ranker_rolling_detail={output_path}")
    print(f"saved_window_ranker_rolling_summary={aggregate_path}")


if __name__ == "__main__":
    main()
