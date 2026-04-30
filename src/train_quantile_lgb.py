from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, align_feature_frame, build_features, fit_history_stats
from .nwp_features import load_or_build_nwp_features, merge_nwp_features
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


def _quantile_params(alpha: float, params: Dict[str, object], seed: int) -> Dict[str, object]:
    out = params_for_seed(params, seed)
    out["objective"] = "quantile"
    out["metric"] = "quantile"
    out["alpha"] = alpha
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM quantile models for uncertainty-aware price prediction.")
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--test-feature", default="")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--nwp-dir", default="")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features_train.csv")
    parser.add_argument("--test-nwp-cache", default="outputs/nwp_features_all.csv")
    parser.add_argument("--model-prefix", default="outputs/quantile_lgb_model")
    parser.add_argument("--metadata-output", default="outputs/quantile_lgb_metadata.json")
    parser.add_argument("--val-pred-output", default="outputs/val_predictions_quantile_lgb.csv")
    parser.add_argument("--prediction-output", default="outputs/test_predictions_quantile_lgb.csv")
    parser.add_argument("--submission-output", default="outputs/output_quantile_lgb.csv")
    parser.add_argument("--meta-output", default="outputs/quantile_lgb_strategy_meta.csv")
    parser.add_argument("--feature-importance-output", default="outputs/feature_importance_quantile_lgb.csv")
    parser.add_argument("--quantiles", default="0.1,0.5,0.9")
    parser.add_argument("--threshold-grid", default="0,500,1000,2000,5000,10000")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-days", type=int, default=0)
    parser.add_argument("--val-start-date", default="")
    parser.add_argument("--val-end-date", default="")
    parser.add_argument("--seeds", default="42")
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

    base_params = DEFAULT_PARAMS.copy()
    if args.params_json:
        base_params.update(json.loads(args.params_json))
    seeds = parse_seeds(args.seeds)
    quantiles = [float(item.strip()) for item in args.quantiles.split(",") if item.strip()]
    if 0.5 not in quantiles:
        quantiles.append(0.5)
    quantiles = sorted(set(quantiles))

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
    train_y = train_df[args.target_col].to_numpy(dtype=float)
    val_y = val_df[args.target_col].to_numpy(dtype=float)

    val_quantile_preds: Dict[float, np.ndarray] = {}
    best_iterations: Dict[float, List[int]] = {}
    for alpha in quantiles:
        preds = []
        rounds_for_alpha = []
        for seed in seeds:
            model = train_booster(
                train_x,
                train_y,
                val_x,
                val_y,
                params=_quantile_params(alpha, base_params, seed),
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            best_iteration = int(model.best_iteration or args.num_boost_round)
            rounds_for_alpha.append(best_iteration)
            preds.append(model.predict(val_x, num_iteration=best_iteration))
        val_quantile_preds[alpha] = np.mean(np.vstack(preds), axis=0)
        best_iterations[alpha] = rounds_for_alpha

    median_pred = val_quantile_preds[0.5]
    val_pred_df = pd.DataFrame({TIME_COL: val_df[TIME_COL].to_numpy(), args.target_col: val_y, "pred_price": median_pred})
    for alpha, pred in val_quantile_preds.items():
        val_pred_df[f"pred_q{int(alpha * 100):02d}"] = pred
    if 0.1 in val_quantile_preds and 0.9 in val_quantile_preds:
        val_pred_df["pred_q90_q10_width"] = val_quantile_preds[0.9] - val_quantile_preds[0.1]
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
    threshold_summary.to_csv(Path(args.val_pred_output).with_name("threshold_search_quantile_lgb.csv"), index=False)
    print(f"quantile_median_rmse={rmse(val_y, median_pred):.6f}")
    print(f"quantile_median_mae={mae(val_y, median_pred):.6f}")
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
    full_y = df[args.target_col].to_numpy(dtype=float)

    model_paths: Dict[str, List[str]] = {}
    final_models: Dict[float, List[object]] = {}
    importances = []
    for alpha in quantiles:
        alpha_key = f"q{int(alpha * 100):02d}"
        model_paths[alpha_key] = []
        final_models[alpha] = []
        for seed, rounds in zip(seeds, best_iterations[alpha]):
            model = train_booster(
                full_x,
                full_y,
                None,
                None,
                params=_quantile_params(alpha, base_params, seed),
                num_boost_round=rounds,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            model_output = f"{args.model_prefix}_{alpha_key}.txt"
            model_path = model_path_for_seed(model_output, seed, len(seeds))
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save_model(model_path)
            model_paths[alpha_key].append(model_path)
            final_models[alpha].append(model)
            if alpha == 0.5:
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
            "quantiles": quantiles,
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
                "rmse": rmse(val_y, median_pred),
                "mae": mae(val_y, median_pred),
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
        test_quantiles = {}
        for alpha, models in final_models.items():
            test_quantiles[alpha] = np.mean(
                np.vstack(
                    [
                        model.predict(test_x, num_iteration=rounds)
                        for model, rounds in zip(models, best_iterations[alpha])
                    ]
                ),
                axis=0,
            )
        pred_out = pd.DataFrame({TIME_COL: test_df[TIME_COL].to_numpy(), "鐎圭偞妞傛禒閿嬬壐": test_quantiles[0.5]})
        for alpha, pred in test_quantiles.items():
            pred_out[f"pred_q{int(alpha * 100):02d}"] = pred
        if 0.1 in test_quantiles and 0.9 in test_quantiles:
            pred_out["pred_q90_q10_width"] = test_quantiles[0.9] - test_quantiles[0.1]
        Path(args.prediction_output).parent.mkdir(parents=True, exist_ok=True)
        pred_out.to_csv(args.prediction_output, index=False)
        submission, meta = generate_strategy(
            pred_out[[TIME_COL, "鐎圭偞妞傛禒閿嬬壐"]],
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
