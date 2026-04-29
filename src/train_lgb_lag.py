from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, align_feature_frame, build_features, fit_history_stats
from .lag_features import add_training_lag_features, all_lag_feature_names, recursive_lag_feature_frame
from .storage_optimizer import generate_strategy
from .train_lgb import DEFAULT_PARAMS, load_training_frame, mae, params_for_seed, rmse, train_booster
from .validate_profit import search_best_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM with recursive price lag features.")
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--test-feature", required=True)
    parser.add_argument("--model-output", default="outputs/lgb_lag_model.txt")
    parser.add_argument("--metadata-output", default="outputs/lgb_lag_metadata.json")
    parser.add_argument("--prediction-output", default="outputs/test_predictions_lag.csv")
    parser.add_argument("--submission-output", default="outputs/output_lag_recursive.csv")
    parser.add_argument("--val-days", type=int, default=59)
    parser.add_argument("--seeds", default="42,2024,2026")
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--threshold-grid", default="0,2000,5000,10000,20000,30000,50000")
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed; run: pip install -r requirements.txt") from exc

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    df = load_training_frame(args.train_feature, args.train_label, target_col=TARGET_COL)
    df = add_training_lag_features(df, target_col=TARGET_COL)

    dates = sorted(df[TIME_COL].dt.date.unique())
    val_dates = set(dates[-args.val_days :])
    train_df = df[~df[TIME_COL].dt.date.isin(val_dates)].copy()
    val_df = df[df[TIME_COL].dt.date.isin(val_dates)].copy()

    stats = fit_history_stats(train_df, target_col=TARGET_COL)
    train_features = build_features(train_df, history_stats=stats)
    val_features = build_features(val_df, history_stats=stats)
    feature_columns = train_features.feature_columns

    train_x = align_feature_frame(train_features.frame, feature_columns)
    val_x_true_lag = align_feature_frame(val_features.frame, feature_columns)
    train_y = train_df[TARGET_COL].to_numpy(dtype=float)
    val_y = val_df[TARGET_COL].to_numpy(dtype=float)

    val_preds_true_lag: List[np.ndarray] = []
    models = []
    best_iterations = []
    for seed in seeds:
        model = train_booster(
            train_x,
            train_y,
            val_x_true_lag,
            val_y,
            params=params_for_seed(DEFAULT_PARAMS, seed),
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        best_iteration = int(model.best_iteration or args.num_boost_round)
        models.append(model)
        best_iterations.append(best_iteration)
        val_preds_true_lag.append(model.predict(val_x_true_lag, num_iteration=best_iteration))

    val_pred = np.mean(np.vstack(val_preds_true_lag), axis=0)
    val_pred_df = pd.DataFrame({TIME_COL: val_df[TIME_COL].to_numpy(), TARGET_COL: val_y, "pred_price": val_pred})
    best_threshold, threshold_summary = search_best_threshold(
        val_pred_df,
        [float(x) for x in args.threshold_grid.split(",") if x.strip()],
        pred_col="pred_price",
        true_col=TARGET_COL,
    )

    print(f"true_lag_validation_rmse={rmse(val_y, val_pred):.6f}")
    print(f"true_lag_validation_mae={mae(val_y, val_pred):.6f}")
    print(threshold_summary.to_string(index=False))

    full_df = add_training_lag_features(load_training_frame(args.train_feature, args.train_label), target_col=TARGET_COL)
    full_stats = fit_history_stats(full_df, target_col=TARGET_COL)
    full_features = build_features(full_df, history_stats=full_stats)
    full_feature_columns = full_features.feature_columns
    full_x = align_feature_frame(full_features.frame, full_feature_columns)
    full_y = full_df[TARGET_COL].to_numpy(dtype=float)

    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    final_models = []
    model_paths = []
    for seed, rounds in zip(seeds, best_iterations):
        model = train_booster(
            full_x,
            full_y,
            None,
            None,
            params=params_for_seed(DEFAULT_PARAMS, seed),
            num_boost_round=rounds,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        path = Path(args.model_output)
        model_path = str(path.with_name(f"{path.stem}_seed{seed}{path.suffix}"))
        model.save_model(model_path)
        final_models.append(model)
        model_paths.append(model_path)

    test_df = pd.read_csv(args.test_feature)
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
    history_prices = full_df[[TIME_COL, TARGET_COL]].copy()

    def feature_builder(frame: pd.DataFrame) -> pd.DataFrame:
        feat = build_features(frame, history_stats=full_stats)
        return align_feature_frame(feat.frame, full_feature_columns)

    def predict_fn(x: pd.DataFrame) -> float:
        preds = [model.predict(x, num_iteration=rounds)[0] for model, rounds in zip(final_models, best_iterations)]
        return float(np.mean(preds))

    test_pred = recursive_lag_feature_frame(test_df, history_prices, predict_fn, feature_builder)
    pred_out = pd.DataFrame({TIME_COL: test_df[TIME_COL].to_numpy(), "实时价格": test_pred})
    Path(args.prediction_output).parent.mkdir(parents=True, exist_ok=True)
    pred_out.to_csv(args.prediction_output, index=False)

    submission, meta = generate_strategy(pred_out, threshold=float(best_threshold["threshold"]))
    submission.to_csv(args.submission_output, index=False)
    meta.to_csv(Path(args.submission_output).with_name("lag_recursive_strategy_meta.csv"), index=False)

    Path(args.metadata_output).write_text(
        json.dumps(
            {
                "model_paths": model_paths,
                "feature_columns": full_feature_columns,
                "lag_features": all_lag_feature_names(),
                "best_threshold": float(best_threshold["threshold"]),
                "validation": {
                    "true_lag_rmse": rmse(val_y, val_pred),
                    "true_lag_mae": mae(val_y, val_pred),
                    "avg_profit": float(best_threshold["avg_profit"]),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved_submission={args.submission_output}, rows={len(submission)}")


if __name__ == "__main__":
    main()

