from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, align_feature_frame, build_features, fit_history_stats
from .nwp_features import load_or_build_nwp_features, merge_nwp_features
from .storage_optimizer import generate_strategy
from .train_lgb import (
    load_training_frame,
    mae,
    model_path_for_seed,
    params_for_seed,
    parse_seeds,
    rmse,
    save_feature_importance,
    save_metadata,
    split_by_day,
)
from .validate_profit import parse_threshold_grid, search_best_threshold


RANKER_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "min_data_in_leaf": 32,
    "label_gain": list(range(16)),
    "seed": 42,
    "verbose": -1,
}


def make_rank_labels(df: pd.DataFrame, target_col: str = TARGET_COL, bins: int = 16) -> np.ndarray:
    work = df[[TIME_COL, target_col]].copy().reset_index(drop=True)
    work[TIME_COL] = pd.to_datetime(work[TIME_COL])
    labels = np.zeros(len(work), dtype=int)
    for _, index in work.groupby(work[TIME_COL].dt.date, sort=False).groups.items():
        prices = work.loc[index, target_col]
        pct = prices.rank(method="first", pct=True).to_numpy(dtype=float)
        labels[np.asarray(index)] = np.clip(np.floor(pct * bins), 0, bins - 1).astype(int)
    return labels


def make_groups(df: pd.DataFrame) -> List[int]:
    dates = pd.to_datetime(df[TIME_COL]).dt.date
    return [int(size) for size in df.groupby(dates, sort=False).size().to_list()]


def train_ranker(
    train_x: pd.DataFrame,
    train_label: np.ndarray,
    train_group: List[int],
    val_x: pd.DataFrame,
    val_label: np.ndarray,
    val_group: List[int],
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed; run: pip install -r requirements.txt") from exc

    train_set = lgb.Dataset(train_x, label=train_label, group=train_group, feature_name=list(train_x.columns))
    val_set = lgb.Dataset(val_x, label=val_label, group=val_group, feature_name=list(train_x.columns), reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(100)],
    )


def _fit_and_predict(
    train_x: pd.DataFrame,
    train_label: np.ndarray,
    train_group: List[int],
    val_x: pd.DataFrame,
    val_label: np.ndarray,
    val_group: List[int],
    seeds: List[int],
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Tuple[List[object], List[int], np.ndarray]:
    models = []
    best_iterations = []
    val_preds = []
    for seed in seeds:
        model = train_ranker(
            train_x,
            train_label,
            train_group,
            val_x,
            val_label,
            val_group,
            params=params_for_seed(params, seed),
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        best_iteration = int(model.best_iteration or num_boost_round)
        models.append(model)
        best_iterations.append(best_iteration)
        val_preds.append(model.predict(val_x, num_iteration=best_iteration))
    return models, best_iterations, np.mean(np.vstack(val_preds), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LightGBM ranker for daily price-window selection.")
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--test-feature", default="")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--nwp-dir", default="")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features_train.csv")
    parser.add_argument("--test-nwp-cache", default="outputs/nwp_features_all.csv")
    parser.add_argument("--model-output", default="outputs/lgb_ranker_model.txt")
    parser.add_argument("--metadata-output", default="outputs/lgb_ranker_metadata.json")
    parser.add_argument("--val-pred-output", default="outputs/val_predictions_lgb_ranker.csv")
    parser.add_argument("--prediction-output", default="outputs/test_predictions_lgb_ranker.csv")
    parser.add_argument("--submission-output", default="outputs/output_lgb_ranker.csv")
    parser.add_argument("--meta-output", default="outputs/lgb_ranker_strategy_meta.csv")
    parser.add_argument("--feature-importance-output", default="outputs/feature_importance_lgb_ranker.csv")
    parser.add_argument("--threshold-grid", default="0,500,1000,2000,5000,10000")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-days", type=int, default=0)
    parser.add_argument("--val-start-date", default="")
    parser.add_argument("--val-end-date", default="")
    parser.add_argument("--seeds", default="42,2024,2026")
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--use-exact-calendar-history", action="store_true")
    parser.add_argument("--use-forecast-bias", action="store_true")
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--params-json", default="")
    args = parser.parse_args()

    params = RANKER_PARAMS.copy()
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
    stats = fit_history_stats(train_df, target_col=args.target_col)
    train_features = build_features(
        train_df,
        history_stats=stats,
        use_exact_calendar_history=args.use_exact_calendar_history,
        use_forecast_bias=args.use_forecast_bias,
    )
    val_features = build_features(
        val_df,
        history_stats=stats,
        use_exact_calendar_history=args.use_exact_calendar_history,
        use_forecast_bias=args.use_forecast_bias,
    )
    feature_columns = train_features.feature_columns
    train_x = align_feature_frame(train_features.frame, feature_columns)
    val_x = align_feature_frame(val_features.frame, feature_columns)
    train_label = make_rank_labels(train_df, target_col=args.target_col)
    val_label = make_rank_labels(val_df, target_col=args.target_col)
    train_group = make_groups(train_df)
    val_group = make_groups(val_df)

    _, best_iterations, val_score = _fit_and_predict(
        train_x,
        train_label,
        train_group,
        val_x,
        val_label,
        val_group,
        seeds,
        params,
        args.num_boost_round,
        args.early_stopping_rounds,
    )

    val_y = val_df[args.target_col].to_numpy(dtype=float)
    val_pred_df = pd.DataFrame({TIME_COL: val_df[TIME_COL].to_numpy(), args.target_col: val_y, "pred_price": val_score})
    Path(args.val_pred_output).parent.mkdir(parents=True, exist_ok=True)
    val_pred_df.to_csv(args.val_pred_output, index=False)
    best_threshold, threshold_summary = search_best_threshold(
        val_pred_df,
        parse_threshold_grid(args.threshold_grid),
        pred_col="pred_price",
        true_col=args.target_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
    )
    threshold_summary.to_csv(Path(args.val_pred_output).with_name("threshold_search_lgb_ranker.csv"), index=False)
    print(f"ranker_score_rmse_against_price={rmse(val_y, val_score):.6f}")
    print(f"ranker_score_mae_against_price={mae(val_y, val_score):.6f}")
    print(threshold_summary.to_string(index=False))

    full_stats = fit_history_stats(df, target_col=args.target_col)
    full_features = build_features(
        df,
        history_stats=full_stats,
        use_exact_calendar_history=args.use_exact_calendar_history,
        use_forecast_bias=args.use_forecast_bias,
    )
    full_feature_columns = full_features.feature_columns
    full_x = align_feature_frame(full_features.frame, full_feature_columns)
    full_label = make_rank_labels(df, target_col=args.target_col)
    full_group = make_groups(df)

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed; run: pip install -r requirements.txt") from exc

    model_paths: List[str] = []
    final_models = []
    importances = []
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    for seed, rounds in zip(seeds, best_iterations):
        train_set = lgb.Dataset(full_x, label=full_label, group=full_group, feature_name=list(full_x.columns))
        model = lgb.train(params_for_seed(params, seed), train_set, num_boost_round=rounds, callbacks=[lgb.log_evaluation(100)])
        model_path = model_path_for_seed(args.model_output, seed, len(seeds))
        model.save_model(model_path)
        model_paths.append(model_path)
        final_models.append(model)
        importances.append(model.feature_importance(importance_type="gain"))

    save_feature_importance(args.feature_importance_output, full_feature_columns, importances)
    save_metadata(
        args.metadata_output,
        {
            "target_col": args.target_col,
            "feature_columns": full_feature_columns,
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
                "avg_profit": float(best_threshold["avg_profit"]),
                "capture_ratio": float(best_threshold.get("capture_ratio", 0.0)),
                "window_hit_rate": float(best_threshold.get("window_hit_rate", 0.0)),
            },
        },
    )

    if args.test_feature:
        test_df = pd.read_csv(args.test_feature)
        test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
        if args.nwp_dir:
            nwp = load_or_build_nwp_features(
                args.nwp_dir,
                args.test_nwp_cache,
                start_time=str(test_df[TIME_COL].min()),
                end_time=str(test_df[TIME_COL].max()),
            )
            test_df = merge_nwp_features(test_df, nwp)
        test_features = build_features(
            test_df,
            history_stats=full_stats,
            use_exact_calendar_history=args.use_exact_calendar_history,
            use_forecast_bias=args.use_forecast_bias,
        )
        test_x = align_feature_frame(test_features.frame, full_feature_columns)
        test_score = np.mean(
            np.vstack(
                [
                    model.predict(test_x, num_iteration=rounds)
                    for model, rounds in zip(final_models, best_iterations)
                ]
            ),
            axis=0,
        )
        pred_out = pd.DataFrame({TIME_COL: test_df[TIME_COL].to_numpy(), "鐎圭偞妞傛禒閿嬬壐": test_score})
        Path(args.prediction_output).parent.mkdir(parents=True, exist_ok=True)
        pred_out.to_csv(args.prediction_output, index=False)
        submission, meta = generate_strategy(
            pred_out,
            threshold=float(best_threshold["threshold"]),
            charge_start_min=args.charge_start_min,
            charge_start_max=args.charge_start_max,
            discharge_start_min=args.discharge_start_min,
            discharge_start_max=args.discharge_start_max,
        )
        submission.to_csv(args.submission_output, index=False)
        meta.to_csv(args.meta_output, index=False)
        print(f"saved_submission={args.submission_output}, rows={len(submission)}")


if __name__ == "__main__":
    main()
