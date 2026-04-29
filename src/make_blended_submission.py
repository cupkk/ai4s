from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL
from .make_seasonal_submission import complete_daily_prices
from .storage_optimizer import generate_strategy


def fit_month_slot_prior(label_df: pd.DataFrame, target_col: str) -> Dict[int, np.ndarray]:
    complete = complete_daily_prices(label_df, target_col=target_col)
    priors: Dict[int, np.ndarray] = {}
    for month, group in complete.groupby("month", sort=True):
        prior = (
            group.groupby("slot")[target_col]
            .mean()
            .reindex(range(96))
            .interpolate()
            .ffill()
            .bfill()
            .to_numpy(dtype=float)
        )
        priors[int(month)] = prior
    return priors


def add_blended_price(
    price_df: pd.DataFrame,
    priors: Dict[int, np.ndarray],
    alpha: float,
    price_col: str = "实时价格",
) -> pd.DataFrame:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    df = price_df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    if price_col not in df.columns:
        raise ValueError(f"price column not found: {price_col}")
    df["month"] = df[TIME_COL].dt.month
    df["slot"] = df[TIME_COL].dt.hour * 4 + df[TIME_COL].dt.minute // 15
    global_prior = np.mean(np.vstack(list(priors.values())), axis=0)
    prior_values = []
    for month, slot in zip(df["month"].to_numpy(), df["slot"].to_numpy()):
        prior_values.append(priors.get(int(month), global_prior)[int(slot)])
    df["seasonal_prior_price"] = prior_values
    df["strategy_price"] = alpha * df[price_col].astype(float) + (1.0 - alpha) * df[
        "seasonal_prior_price"
    ].astype(float)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strategy from model/seasonal blended prices.")
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--price-csv", required=True)
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--price-col", default="实时价格")
    parser.add_argument("--alpha", type=float, default=0.35, help="0=seasonal only, 1=model only")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--charge-start-min", type=int, default=51)
    parser.add_argument("--charge-start-max", type=int, default=55)
    parser.add_argument("--discharge-start-min", type=int, default=66)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--meta-output", default="outputs/blended_strategy_meta.csv")
    args = parser.parse_args()

    label_df = pd.read_csv(args.train_label)
    price_df = pd.read_csv(args.price_csv)
    priors = fit_month_slot_prior(label_df, target_col=args.target_col)
    strategy_df = add_blended_price(price_df, priors, alpha=args.alpha, price_col=args.price_col)
    out, meta = generate_strategy(
        strategy_df[[TIME_COL, "strategy_price"]],
        threshold=args.threshold,
        price_col="strategy_price",
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
    )
    # Keep the model price in the submission's price column; only power is scored,
    # and the model price is more informative for inspection than the blended score.
    out["实时价格"] = price_df[args.price_col].astype(float).to_numpy()
    out = out[["times", "实时价格", "power"]]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    if args.meta_output:
        Path(args.meta_output).parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(args.meta_output, index=False)
    print(f"saved_submission={args.output}, rows={len(out)}, alpha={args.alpha}")


if __name__ == "__main__":
    main()

