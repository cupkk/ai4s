from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, align_feature_frame, build_features, fit_history_stats
from .nwp_features import load_or_build_nwp_features, merge_nwp_features
from .price_history_features import add_price_history_features, fit_price_history_features
from .storage_optimizer import generate_strategy
from .train_lgb import (
    DEFAULT_PARAMS,
    load_training_frame,
    mae,
    model_path_for_seed,
    params_for_seed,
    parse_seeds,
    rmse,
    save_feature_importance,
    save_metadata,
    split_by_day,
    train_booster,
)
from .validate_profit import parse_threshold_grid, search_best_threshold


def _base_prediction(frame: pd.DataFrame, preferred_col: str) -> np.ndarray:
    if preferred_col in frame.columns:
        return frame[preferred_col].to_numpy(dtype=float)
    if "hist_slot_mean" in frame.columns:
        return frame["hist_slot_mean"].to_numpy(dtype=float)
    raise ValueError("residual base column is unavailable; history features were not built")


def _predict_ensemble(models, model_paths: List[str], x: pd.DataFrame, iterations: List[int] | None = None) -> np.ndarray:
    if models:
        preds = []
        for idx, model in enumerate(models):
            num_iteration = None if iterations is None else iterations[idx]
            preds.append(model.predict(x, num_iteration=num_iteration))
        return np.mean(np.vstack(preds), axis=0)

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed; run: pip install -r requirements.txt") from exc
    preds = [lgb.Booster(model_file=path).predict(x) for path in model_paths]
    return np.mean(np.vstack(preds), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LightGBM residual model over historical price priors.")
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--test-feature", default="")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--nwp-dir", default="")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features_train.csv")
    parser.add_argument("--test-nwp-cache", default="outputs/nwp_features_all.csv")
    parser.add_argument("--model-output", default="outputs/residual_lgb_model.txt")
    parser.add_argument("--metadata-output", default="outputs/residual_lgb_metadata.json")
    parser.add_argument("--val-pred-output", default="outputs/val_predictions_residual_lgb.csv")
    parser.add_argument("--prediction-output", default="outputs/test_predictions_residual_lgb.csv")
    parser.add_argument("--submission-output", default="outputs/output_residual_lgb.csv")
    parser.add_argument("--meta-output", default="outputs/residual_lgb_strategy_meta.csv")
    parser.add_argument("--feature-importance-output", default="outputs/feature_importance_residual_lgb.csv")
    parser.add_argument("--threshold-grid", default="0,500,1000,2000,5000,10000,20000")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-days", type=int, default=0)
    parser.add_argument("--val-start-date", default="")
    parser.add_argument("--val-end-date", default="")
    parser.add_argument("--seeds", default="42,2024,2026")
    parser.add_argument("--residual-base-col", default="hist_month_slot_mean")
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
    price_history_stats = fit_price_history_features(train_df, target_col=args.target_col)
    train_model_df = add_price_history_features(train_df, price_history_stats)
    val_model_df = add_price_history_features(val_df, price_history_stats)
    train_stats = fit_history_stats(train_model_df, target_col=args.target_col)
    train_features = build_features(
        train_model_df,
        history_stats=train_stats,
        use_exact_calendar_history=args.use_exact_calendar_history,
        use_forecast_bias=args.use_forecast_bias,
    )
    val_features = build_features(
        val_model_df,
        history_stats=train_stats,
        use_exact_calendar_history=args.use_exact_calendar_history,
        use_forecast_bias=args.use_forecast_bias,
    )
    feature_columns = train_features.feature_columns
    train_x = align_feature_frame(train_features.frame, feature_columns)
    val_x = align_feature_frame(val_features.frame, feature_columns)
    train_y = train_df[args.target_col].to_numpy(dtype=float)
    val_y = val_df[args.target_col].to_numpy(dtype=float)
    base_train = _base_prediction(train_features.frame, args.residual_base_col)
    base_val = _base_prediction(val_features.frame, args.residual_base_col)
    train_residual = train_y - base_train

    val_residual_preds: List[np.ndarray] = []
    best_iterations: List[int] = []
    for seed in seeds:
        model = train_booster(
            train_x,
            train_residual,
            val_x,
            val_y - base_val,
            params=params_for_seed(params, seed),
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        best_iteration = int(model.best_iteration or args.num_boost_round)
        best_iterations.append(best_iteration)
        val_residual_preds.append(model.predict(val_x, num_iteration=best_iteration))

    val_pred = base_val + np.mean(np.vstack(val_residual_preds), axis=0)
    val_pred_df = pd.DataFrame({TIME_COL: val_df[TIME_COL].to_numpy(), args.target_col: val_y, "pred_price": val_pred})
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
    threshold_summary.to_csv(Path(args.val_pred_output).with_name("threshold_search_residual_lgb.csv"), index=False)
    print(f"residual_validation_rmse={rmse(val_y, val_pred):.6f}")
    print(f"residual_validation_mae={mae(val_y, val_pred):.6f}")
    print(threshold_summary.to_string(index=False))

    full_price_history_stats = fit_price_history_features(df, target_col=args.target_col)
    full_model_df = add_price_history_features(df, full_price_history_stats)
    full_stats = fit_history_stats(full_model_df, target_col=args.target_col)
    full_features = build_features(
        full_model_df,
        history_stats=full_stats,
        use_exact_calendar_history=args.use_exact_calendar_history,
        use_forecast_bias=args.use_forecast_bias,
    )
    full_feature_columns = full_features.feature_columns
    full_x = align_feature_frame(full_features.frame, full_feature_columns)
    full_y = df[args.target_col].to_numpy(dtype=float)
    full_base = _base_prediction(full_features.frame, args.residual_base_col)

    model_paths: List[str] = []
    final_models = []
    importances: List[np.ndarray] = []
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    for seed, rounds in zip(seeds, best_iterations):
        model = train_booster(
            full_x,
            full_y - full_base,
            None,
            None,
            params=params_for_seed(params, seed),
            num_boost_round=rounds,
            early_stopping_rounds=args.early_stopping_rounds,
        )
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
            "price_history_stats": full_price_history_stats,
            "model_paths": model_paths,
            "seeds": seeds,
            "best_threshold": float(best_threshold["threshold"]),
            "residual_base_col": args.residual_base_col,
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
        test_df = add_price_history_features(test_df, full_price_history_stats)
        test_features = build_features(
            test_df,
            history_stats=full_stats,
            use_exact_calendar_history=args.use_exact_calendar_history,
            use_forecast_bias=args.use_forecast_bias,
        )
        test_x = align_feature_frame(test_features.frame, full_feature_columns)
        test_base = _base_prediction(test_features.frame, args.residual_base_col)
        test_pred = test_base + _predict_ensemble(final_models, model_paths, test_x, best_iterations)
        pred_out = pd.DataFrame({TIME_COL: test_df[TIME_COL].to_numpy(), "鐎圭偞妞傛禒閿嬬壐": test_pred})
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
