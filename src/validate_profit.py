from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .storage_optimizer import evaluate_power, generate_strategy, infer_price_column


def parse_threshold_grid(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("threshold grid is empty")
    return values


def backtest_predictions(
    df: pd.DataFrame,
    threshold: float,
    pred_col: str = "pred_price",
    true_col: str = "A",
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
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
    )
    joined = strategy.merge(df_complete[["times", true_col]], on="times", how="left")
    if joined[true_col].isna().any():
        raise ValueError("strategy times could not be fully matched to true prices")

    joined["__date__"] = pd.to_datetime(joined["times"]).dt.date
    day_rows = []
    for date, group in joined.groupby("__date__", sort=True):
        profit = evaluate_power(group[true_col], group["power"])
        traded = bool((group["power"] != 0).any())
        day_rows.append({"date": str(date), "profit": profit, "traded": traded})

    day_df = pd.DataFrame(day_rows)
    summary = {
        "threshold": float(threshold),
        "days": int(len(day_df)),
        "avg_profit": float(day_df["profit"].mean()),
        "total_profit": float(day_df["profit"].sum()),
        "traded_days": int(day_df["traded"].sum()),
        "loss_days": int((day_df["profit"] < 0).sum()),
        "skipped_incomplete_days": skipped_days,
    }
    meta = meta.rename(columns={"traded": "pred_traded"})
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


if __name__ == "__main__":
    main()
