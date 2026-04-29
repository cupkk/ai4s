from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL
from .storage_optimizer import evaluate_power, optimize_one_day


def complete_daily_prices(label_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = label_df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["date"] = df[TIME_COL].dt.date
    df["month"] = df[TIME_COL].dt.month
    df["slot"] = df[TIME_COL].dt.hour * 4 + df[TIME_COL].dt.minute // 15

    parts = []
    for _, group in df.groupby("date", sort=True):
        group = group.sort_values(TIME_COL)
        if len(group) == 96 and sorted(group["slot"].tolist()) == list(range(96)):
            parts.append(group)
    if not parts:
        raise ValueError("no complete 96-point days found in training label file")
    return pd.concat(parts, ignore_index=True)


def fit_monthly_windows(
    label_df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Dict[int, Tuple[int, int, float]]:
    complete = complete_daily_prices(label_df, target_col=target_col)
    windows: Dict[int, Tuple[int, int, float]] = {}
    for month, group in complete.groupby("month", sort=True):
        avg_price = (
            group.groupby("slot")[target_col]
            .mean()
            .reindex(range(96))
            .interpolate()
            .ffill()
            .bfill()
            .to_numpy()
        )
        result = optimize_one_day(avg_price, threshold=-1e18)
        if result.charge_start is None or result.discharge_start is None:
            raise RuntimeError(f"failed to fit seasonal window for month={month}")
        windows[int(month)] = (
            int(result.charge_start),
            int(result.discharge_start),
            float(result.best_profit),
        )
    return windows


def make_seasonal_strategy(
    test_price_df: pd.DataFrame,
    windows: Dict[int, Tuple[int, int, float]],
    price_col: str = "实时价格",
    fallback_month: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = test_price_df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    if price_col not in df.columns:
        df[price_col] = 0.0
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    df["date"] = df[TIME_COL].dt.date
    df["month"] = df[TIME_COL].dt.month

    outputs = []
    meta = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values(TIME_COL)
        if len(group) != 96:
            raise ValueError(f"{date} must contain 96 rows, got {len(group)}")
        month = int(group["month"].iloc[0])
        tc, td, fit_profit = windows.get(month, windows[fallback_month])
        power = np.zeros(96)
        power[tc : tc + 8] = -1000
        power[td : td + 8] = 1000
        outputs.append(
            pd.DataFrame(
                {
                    "times": group[TIME_COL].to_numpy(),
                    "实时价格": group[price_col].to_numpy(dtype=float),
                    "power": power,
                }
            )
        )
        meta.append(
            {
                "date": str(date),
                "month": month,
                "charge_start": tc,
                "discharge_start": td,
                "fit_window_profit": fit_profit,
                "predicted_price_profit": evaluate_power(group[price_col], power),
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a stable monthly seasonal strategy.")
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--price-csv", required=True, help="CSV with test times and predicted price.")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--price-col", default="实时价格")
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--meta-output", default="outputs/seasonal_strategy_meta.csv")
    args = parser.parse_args()

    label_df = pd.read_csv(args.train_label)
    price_df = pd.read_csv(args.price_csv)
    windows = fit_monthly_windows(label_df, target_col=args.target_col)
    out, meta = make_seasonal_strategy(price_df, windows, price_col=args.price_col)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    if args.meta_output:
        Path(args.meta_output).parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(args.meta_output, index=False)
    print(f"saved_submission={args.output}, rows={len(out)}, windows={windows}")


if __name__ == "__main__":
    main()

