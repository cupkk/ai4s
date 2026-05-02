from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, align_feature_frame, build_features, fit_history_stats
from .nwp_features import load_or_build_nwp_features, merge_nwp_features
from .price_history_features import add_price_history_features, fit_price_history_features
from .storage_optimizer import evaluate_power, optimize_one_day
from .train_lgb import (
    DEFAULT_PARAMS,
    load_training_frame,
    params_for_seed,
    parse_seeds,
    split_by_day,
    train_booster,
)


DEFAULT_POINT_FEATURES = [
    "price_hist_same_month_day_slot",
    "price_hist_month_slot_median",
    "price_hist_month_slot_p10",
    "price_hist_month_slot_p90",
    "price_hist_recent_28d_slot_mean",
    "price_hist_recent_28d_slot_std",
    "hist_slot_mean",
    "hist_month_slot_mean",
    "net_load",
    "renewable_ratio",
    "supply_margin",
    "net_load_intertie_minus",
    "net_load_intertie_plus",
    "nwp_ghi_mean",
    "nwp_wind_speed_mean",
    "nwp_wind_speed_cube_mean",
    "nwp_t2m_mean",
    "nwp_tcc_mean",
]


def _block_values(values: np.ndarray, block_size: int, reducer: str) -> np.ndarray:
    out = []
    for start in range(len(values) - block_size + 1):
        window = values[start : start + block_size]
        if reducer == "sum":
            out.append(float(np.sum(window)))
        elif reducer == "mean":
            out.append(float(np.mean(window)))
        else:
            raise ValueError(f"unsupported reducer: {reducer}")
    return np.asarray(out, dtype=float)


def _candidate_frame_for_day(
    day_features: pd.DataFrame,
    true_prices: np.ndarray | None,
    point_features: Sequence[str],
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
    block_size: int = 8,
    power_value: float = 1000.0,
) -> pd.DataFrame:
    if len(day_features) != 96:
        raise ValueError(f"expected 96 rows per day, got {len(day_features)}")
    max_charge_start = 96 - 2 * block_size
    max_start = 96 - block_size
    c_min = max(0, int(charge_start_min))
    c_max = min(max_charge_start, int(charge_start_max))
    d_min = max(block_size, int(discharge_start_min))
    d_max = min(max_start, int(discharge_start_max))

    block_means: Dict[str, np.ndarray] = {}
    for col in point_features:
        block_means[col] = _block_values(day_features[col].to_numpy(dtype=float), block_size, "mean")
    if true_prices is not None:
        true_block_sum = _block_values(true_prices, block_size, "sum")

    rows = []
    for tc in range(c_min, c_max + 1):
        for td in range(max(tc + block_size, d_min), d_max + 1):
            row = {
                "charge_start": tc,
                "discharge_start": td,
                "gap_slots": td - tc,
                "charge_hour": tc / 4.0,
                "discharge_hour": td / 4.0,
            }
            for col, values in block_means.items():
                charge_value = values[tc]
                discharge_value = values[td]
                row[f"charge_{col}"] = charge_value
                row[f"discharge_{col}"] = discharge_value
                row[f"spread_{col}"] = discharge_value - charge_value
            if true_prices is not None:
                row["true_window_profit"] = power_value * (true_block_sum[td] - true_block_sum[tc])
            rows.append(row)
    return pd.DataFrame(rows)


def build_window_dataset(
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    point_features: Sequence[str],
    target_col: str,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
    include_target: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df[[TIME_COL] + ([target_col] if include_target else [])].copy()
    work[TIME_COL] = pd.to_datetime(work[TIME_COL])
    features = feature_frame.copy()
    features[TIME_COL] = pd.to_datetime(df[TIME_COL]).to_numpy()
    features["__date__"] = features[TIME_COL].dt.date
    work["__date__"] = work[TIME_COL].dt.date

    candidates = []
    meta_rows = []
    for date, day_idx in features.groupby("__date__", sort=True).groups.items():
        day_features = features.loc[day_idx, list(point_features)].reset_index(drop=True)
        if len(day_features) != 96:
            continue
        true_prices = None
        if include_target:
            true_prices = work.loc[day_idx, target_col].to_numpy(dtype=float)
        day_candidates = _candidate_frame_for_day(
            day_features,
            true_prices=true_prices,
            point_features=point_features,
            charge_start_min=charge_start_min,
            charge_start_max=charge_start_max,
            discharge_start_min=discharge_start_min,
            discharge_start_max=discharge_start_max,
        )
        day_candidates["date"] = str(date)
        candidates.append(day_candidates)
        meta_rows.append({"date": str(date), "candidate_count": len(day_candidates)})
    if not candidates:
        raise ValueError("no complete daily windows available")
    return pd.concat(candidates, ignore_index=True), pd.DataFrame(meta_rows)


def choose_daily_windows(candidate_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    idx = candidate_df.groupby("date", sort=True)[score_col].idxmax()
    return candidate_df.loc[idx].sort_values("date").reset_index(drop=True)


def windows_to_submission(
    base_df: pd.DataFrame,
    selected: pd.DataFrame,
    price_values: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = base_df[[TIME_COL]].copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["__date__"] = df[TIME_COL].dt.date.astype(str)
    if price_values is not None:
        df["__price__"] = price_values
    selected_by_date = {row["date"]: row for row in selected.to_dict("records")}
    outputs = []
    meta = []
    for date, group in df.groupby("__date__", sort=True):
        group = group.sort_values(TIME_COL).reset_index(drop=True)
        if len(group) != 96:
            continue
        row = selected_by_date[date]
        power = np.zeros(96, dtype=float)
        tc = int(row["charge_start"])
        td = int(row["discharge_start"])
        power[tc : tc + 8] = -1000.0
        power[td : td + 8] = 1000.0
        price = (
            np.zeros(96, dtype=float)
            if price_values is None
            else group["__price__"].to_numpy(dtype=float)
        )
        outputs.append(
            pd.DataFrame(
                {
                    "times": group[TIME_COL].to_numpy(),
                    "鐎圭偞妞傛禒閿嬬壐": price,
                    "power": power,
                }
            )
        )
        meta.append(
            {
                "date": date,
                "pred_window_score": float(row["pred_window_profit"]),
                "charge_start": tc,
                "discharge_start": td,
                "traded": True,
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct window-profit model for storage strategy.")
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--test-feature", default="")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--nwp-dir", default="")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features_train.csv")
    parser.add_argument("--test-nwp-cache", default="outputs/nwp_features_all.csv")
    parser.add_argument("--model-output", default="outputs/window_ranker_model.txt")
    parser.add_argument("--metadata-output", default="outputs/window_ranker_metadata.json")
    parser.add_argument("--val-window-output", default="outputs/val_windows_window_ranker.csv")
    parser.add_argument("--submission-output", default="outputs/output_window_ranker.csv")
    parser.add_argument("--meta-output", default="outputs/window_ranker_strategy_meta.csv")
    parser.add_argument("--val-days", type=int, default=59)
    parser.add_argument("--val-start-date", default="")
    parser.add_argument("--val-end-date", default="")
    parser.add_argument("--seeds", default="42,2024,2026")
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--params-json", default="")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    params["objective"] = "regression"
    params["metric"] = "rmse"
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
        val_ratio=0.2,
        val_start_date=args.val_start_date,
        val_end_date=args.val_end_date,
        val_days=args.val_days,
    )

    price_stats = fit_price_history_features(train_df, target_col=args.target_col)
    train_model_df = add_price_history_features(train_df, price_stats)
    val_model_df = add_price_history_features(val_df, price_stats)
    hist_stats = fit_history_stats(train_model_df, target_col=args.target_col)
    train_features = build_features(train_model_df, history_stats=hist_stats)
    val_features = build_features(val_model_df, history_stats=hist_stats)
    point_features = [col for col in DEFAULT_POINT_FEATURES if col in train_features.frame.columns]
    if not point_features:
        raise ValueError("no point features available for window ranker")

    train_windows, _ = build_window_dataset(
        train_df,
        train_features.frame,
        point_features,
        args.target_col,
        args.charge_start_min,
        args.charge_start_max,
        args.discharge_start_min,
        args.discharge_start_max,
        include_target=True,
    )
    val_windows, _ = build_window_dataset(
        val_df,
        val_features.frame,
        point_features,
        args.target_col,
        args.charge_start_min,
        args.charge_start_max,
        args.discharge_start_min,
        args.discharge_start_max,
        include_target=True,
    )
    feature_columns = [
        col for col in train_windows.columns if col not in {"date", "true_window_profit"}
    ]
    train_x = train_windows[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    val_x = val_windows[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_y = train_windows["true_window_profit"].to_numpy(dtype=float)
    val_y = val_windows["true_window_profit"].to_numpy(dtype=float)

    val_preds = []
    best_iterations = []
    for seed in seeds:
        model = train_booster(
            train_x,
            train_y,
            val_x,
            val_y,
            params=params_for_seed(params, seed),
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        best_iteration = int(model.best_iteration or args.num_boost_round)
        best_iterations.append(best_iteration)
        val_preds.append(model.predict(val_x, num_iteration=best_iteration))
    val_windows["pred_window_profit"] = np.mean(np.vstack(val_preds), axis=0)
    selected_val = choose_daily_windows(val_windows, "pred_window_profit")
    val_submission, _ = windows_to_submission(
        val_df,
        selected_val,
        price_values=val_df[args.target_col].to_numpy(dtype=float),
    )
    joined = val_submission.merge(val_df[[TIME_COL, args.target_col]], on=TIME_COL, how="left")
    day_profit = []
    joined["date"] = pd.to_datetime(joined["times"]).dt.date
    for date, group in joined.groupby("date", sort=True):
        oracle = optimize_one_day(
            group[args.target_col].to_numpy(dtype=float),
            charge_start_min=args.charge_start_min,
            charge_start_max=args.charge_start_max,
            discharge_start_min=args.discharge_start_min,
            discharge_start_max=args.discharge_start_max,
        )
        profit = evaluate_power(group[args.target_col], group["power"])
        day_profit.append(
            {
                "date": str(date),
                "profit": profit,
                "oracle_profit": max(0.0, float(oracle.best_profit)),
            }
        )
    day_profit_df = pd.DataFrame(day_profit)
    Path(args.val_window_output).parent.mkdir(parents=True, exist_ok=True)
    selected_val.to_csv(args.val_window_output, index=False)
    print(
        "window_ranker_validation="
        f"avg_profit={day_profit_df['profit'].mean():.6f}, "
        f"capture_ratio={day_profit_df['profit'].sum() / day_profit_df['oracle_profit'].sum():.6f}"
    )

    full_price_stats = fit_price_history_features(df, target_col=args.target_col)
    full_model_df = add_price_history_features(df, full_price_stats)
    full_hist_stats = fit_history_stats(full_model_df, target_col=args.target_col)
    full_features = build_features(full_model_df, history_stats=full_hist_stats)
    full_point_features = [col for col in point_features if col in full_features.frame.columns]
    full_windows, _ = build_window_dataset(
        df,
        full_features.frame,
        full_point_features,
        args.target_col,
        args.charge_start_min,
        args.charge_start_max,
        args.discharge_start_min,
        args.discharge_start_max,
        include_target=True,
    )
    full_feature_columns = [
        col for col in full_windows.columns if col not in {"date", "true_window_profit"}
    ]
    full_x = full_windows[full_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    full_y = full_windows["true_window_profit"].to_numpy(dtype=float)

    final_models = []
    model_paths = []
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    for seed, rounds in zip(seeds, best_iterations):
        model = train_booster(
            full_x,
            full_y,
            None,
            None,
            params=params_for_seed(params, seed),
            num_boost_round=rounds,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        path = Path(args.model_output)
        model_path = str(path.with_name(f"{path.stem}_seed{seed}{path.suffix}"))
        model.save_model(model_path)
        final_models.append(model)
        model_paths.append(model_path)

    Path(args.metadata_output).write_text(
        json.dumps(
            {
                "model_paths": model_paths,
                "seeds": seeds,
                "feature_columns": full_feature_columns,
                "point_features": full_point_features,
                "price_history_stats": full_price_stats,
                "history_stats": full_hist_stats,
                "best_iterations": best_iterations,
                "strategy_constraints": {
                    "charge_start_min": args.charge_start_min,
                    "charge_start_max": args.charge_start_max,
                    "discharge_start_min": args.discharge_start_min,
                    "discharge_start_max": args.discharge_start_max,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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
        test_model_df = add_price_history_features(test_df, full_price_stats)
        test_features = build_features(test_model_df, history_stats=full_hist_stats)
        test_windows, _ = build_window_dataset(
            test_df,
            test_features.frame,
            full_point_features,
            args.target_col,
            args.charge_start_min,
            args.charge_start_max,
            args.discharge_start_min,
            args.discharge_start_max,
            include_target=False,
        )
        test_x = test_windows[full_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_windows["pred_window_profit"] = np.mean(
            np.vstack(
                [
                    model.predict(test_x, num_iteration=rounds)
                    for model, rounds in zip(final_models, best_iterations)
                ]
            ),
            axis=0,
        )
        selected_test = choose_daily_windows(test_windows, "pred_window_profit")
        submission, meta = windows_to_submission(test_df, selected_test)
        submission.to_csv(args.submission_output, index=False)
        meta.to_csv(args.meta_output, index=False)
        print(f"saved_window_ranker_submission={args.submission_output}, rows={len(submission)}")


if __name__ == "__main__":
    main()
