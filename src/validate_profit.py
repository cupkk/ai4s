from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .storage_optimizer import (
    evaluate_power,
    generate_strategy,
    infer_price_column,
    optimize_one_day,
)


def parse_threshold_grid(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("threshold grid is empty")
    return values


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _same_or_near(left: int | None, right: int | None, tolerance: int = 0) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= tolerance


def backtest_predictions(
    df: pd.DataFrame,
    threshold: float,
    pred_col: str = "pred_price",
    true_col: str = "A",
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
    threshold_by_month: Optional[Dict[int, float]] = None,
) -> Tuple[dict, pd.DataFrame]:
    df = df.copy()
    if "times" not in df.columns:
        raise ValueError("missing required column: times")
    df["times"] = pd.to_datetime(df["times"])
    if true_col not in df.columns:
        raise ValueError(f"missing true price column: {true_col}")
    pred_col = infer_price_column(df, pred_col)

    df["__date__"] = df["times"].dt.date
    day_counts = df.groupby("__date__", sort=True).size()
    complete_dates = set(day_counts[day_counts == 96].index)
    skipped_days = int((day_counts != 96).sum())
    if not complete_dates:
        raise ValueError("no complete 96-point validation days available for profit backtest")

    df_complete = df[df["__date__"].isin(complete_dates)].drop(columns=["__date__"]).copy()
    strategy, meta = generate_strategy(
        df_complete[["times", pred_col]].copy(),
        threshold=threshold,
        price_col=pred_col,
        charge_start_min=charge_start_min,
        charge_start_max=charge_start_max,
        discharge_start_min=discharge_start_min,
        discharge_start_max=discharge_start_max,
        threshold_by_month=threshold_by_month,
    )
    joined = strategy.merge(df_complete[["times", true_col]], on="times", how="left")
    if joined[true_col].isna().any():
        raise ValueError("strategy times could not be fully matched to true prices")

    joined["__date__"] = pd.to_datetime(joined["times"]).dt.date
    meta = meta.rename(columns={"traded": "pred_traded"})
    meta_by_date = {row["date"]: row for row in meta.to_dict("records")}
    day_rows = []
    for date, group in joined.groupby("__date__", sort=True):
        profit = evaluate_power(group[true_col], group["power"])
        traded = bool((group["power"] != 0).any())
        oracle = optimize_one_day(
            group[true_col].to_numpy(dtype=float),
            threshold=0.0,
            charge_start_min=charge_start_min,
            charge_start_max=charge_start_max,
            discharge_start_min=discharge_start_min,
            discharge_start_max=discharge_start_max,
        )
        oracle_profit = max(0.0, float(oracle.best_profit))
        oracle_charge_start = oracle.charge_start if oracle.traded else None
        oracle_discharge_start = oracle.discharge_start if oracle.traded else None

        pred_info = meta_by_date.get(str(date), {})
        pred_charge_start = _optional_int(pred_info.get("charge_start"))
        pred_discharge_start = _optional_int(pred_info.get("discharge_start"))
        charge_hit = _same_or_near(pred_charge_start, oracle_charge_start)
        discharge_hit = _same_or_near(pred_discharge_start, oracle_discharge_start)
        charge_hit_2 = _same_or_near(pred_charge_start, oracle_charge_start, tolerance=2)
        discharge_hit_2 = _same_or_near(pred_discharge_start, oracle_discharge_start, tolerance=2)
        day_rows.append(
            {
                "date": str(date),
                "profit": profit,
                "traded": traded,
                "oracle_profit": oracle_profit,
                "regret": oracle_profit - profit,
                "capture_ratio": profit / oracle_profit if oracle_profit > 0 else np.nan,
                "oracle_traded": bool(oracle.traded),
                "oracle_charge_start": oracle_charge_start,
                "oracle_discharge_start": oracle_discharge_start,
                "charge_window_hit": charge_hit,
                "discharge_window_hit": discharge_hit,
                "window_hit": charge_hit and discharge_hit,
                "charge_window_hit_2": charge_hit_2,
                "discharge_window_hit_2": discharge_hit_2,
                "window_hit_2": charge_hit_2 and discharge_hit_2,
            }
        )

    day_df = pd.DataFrame(day_rows)
    oracle_days = day_df["oracle_traded"]
    total_profit = float(day_df["profit"].sum())
    oracle_total_profit = float(day_df["oracle_profit"].sum())
    summary = {
        "threshold": float(threshold),
        "days": int(len(day_df)),
        "avg_profit": float(day_df["profit"].mean()),
        "total_profit": total_profit,
        "avg_oracle_profit": float(day_df["oracle_profit"].mean()),
        "oracle_total_profit": oracle_total_profit,
        "capture_ratio": total_profit / oracle_total_profit if oracle_total_profit > 0 else np.nan,
        "avg_regret": float(day_df["regret"].mean()),
        "total_regret": float(day_df["regret"].sum()),
        "traded_days": int(day_df["traded"].sum()),
        "oracle_traded_days": int(day_df["oracle_traded"].sum()),
        "loss_days": int((day_df["profit"] < 0).sum()),
        "window_hit_rate": float(day_df.loc[oracle_days, "window_hit"].mean())
        if oracle_days.any()
        else np.nan,
        "window_hit_2_rate": float(day_df.loc[oracle_days, "window_hit_2"].mean())
        if oracle_days.any()
        else np.nan,
        "charge_window_hit_rate": float(day_df.loc[oracle_days, "charge_window_hit"].mean())
        if oracle_days.any()
        else np.nan,
        "discharge_window_hit_rate": float(day_df.loc[oracle_days, "discharge_window_hit"].mean())
        if oracle_days.any()
        else np.nan,
        "skipped_incomplete_days": skipped_days,
    }
    day_df = day_df.merge(meta, on="date", how="left")
    return summary, day_df


def search_best_threshold(
    df: pd.DataFrame,
    threshold_grid: Sequence[float],
    pred_col: str = "pred_price",
    true_col: str = "A",
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
) -> Tuple[dict, pd.DataFrame]:
    summaries = []
    for threshold in threshold_grid:
        summary, _ = backtest_predictions(
            df,
            threshold,
            pred_col=pred_col,
            true_col=true_col,
            charge_start_min=charge_start_min,
            charge_start_max=charge_start_max,
            discharge_start_min=discharge_start_min,
            discharge_start_max=discharge_start_max,
        )
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries).sort_values(
        ["avg_profit", "threshold"], ascending=[False, True]
    )
    best = summary_df.iloc[0].to_dict()
    return best, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest storage profit on validation predictions.")
    parser.add_argument("--pred-csv", required=True, help="CSV with times, true price, and predicted price.")
    parser.add_argument("--pred-col", default="pred_price")
    parser.add_argument("--true-col", default="A")
    parser.add_argument("--threshold-grid", default="0,5000,10000,20000,30000,50000,80000,100000")
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--day-output", default="")
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    thresholds = parse_threshold_grid(args.threshold_grid)
    best, summary_df = search_best_threshold(
        df,
        thresholds,
        pred_col=args.pred_col,
        true_col=args.true_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
    )
    print(summary_df.to_string(index=False))
    print(f"best_threshold={best['threshold']}, avg_profit={best['avg_profit']:.6f}")
    if args.summary_output:
        summary_df.to_csv(args.summary_output, index=False)
    if args.day_output:
        _, day_df = backtest_predictions(
            df,
            float(best["threshold"]),
            pred_col=args.pred_col,
            true_col=args.true_col,
            charge_start_min=args.charge_start_min,
            charge_start_max=args.charge_start_max,
            discharge_start_min=args.discharge_start_min,
            discharge_start_max=args.discharge_start_max,
        )
        day_df.to_csv(args.day_output, index=False)


if __name__ == "__main__":
    main()
