from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .storage_optimizer import infer_price_column


def _runsum(values: np.ndarray, block_size: int) -> np.ndarray:
    return np.asarray([values[i : i + block_size].sum() for i in range(len(values) - block_size + 1)])


def _infer_uncertainty(df: pd.DataFrame, preferred: str = "") -> Optional[str]:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"uncertainty column not found: {preferred}")
        return preferred
    if "pred_std" in df.columns:
        return "pred_std"
    seed_cols = [col for col in df.columns if col.startswith("pred_price_seed")]
    if len(seed_cols) >= 2:
        df["pred_std"] = df[seed_cols].std(axis=1)
        return "pred_std"
    return None


def generate_robust_strategy(
    price_df: pd.DataFrame,
    lambda_uncertainty: float,
    threshold: float,
    price_col: str | None = None,
    uncertainty_col: str = "",
    block_size: int = 8,
    power_value: float = 1000.0,
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = price_df.copy()
    if "times" not in df.columns:
        raise ValueError("missing required column: times")
    df["times"] = pd.to_datetime(df["times"])
    df = df.sort_values("times").reset_index(drop=True)
    price_col = infer_price_column(df, price_col)
    uncertainty_col = _infer_uncertainty(df, uncertainty_col) or ""
    if uncertainty_col:
        df[uncertainty_col] = df[uncertainty_col].fillna(0.0).clip(lower=0.0)
    else:
        df["pred_std"] = 0.0
        uncertainty_col = "pred_std"

    outputs = []
    meta = []
    df["__date__"] = df["times"].dt.date
    for date, group in df.groupby("__date__", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        if len(group) != 96:
            raise ValueError(f"{date} must contain 96 rows, got {len(group)}")
        prices = group[price_col].to_numpy(dtype=float)
        uncertainty = group[uncertainty_col].to_numpy(dtype=float)
        price_sum = _runsum(prices, block_size)
        uncertainty_sum = _runsum(uncertainty, block_size)

        max_charge_start = 96 - 2 * block_size
        max_start = 96 - block_size
        c_min = max(0, int(charge_start_min))
        c_max = min(max_charge_start, int(charge_start_max))
        d_min = max(block_size, int(discharge_start_min))
        d_max = min(max_start, int(discharge_start_max))

        best_score = -np.inf
        best_pred_profit = -np.inf
        best_tc = None
        best_td = None
        for tc in range(c_min, c_max + 1):
            for td in range(max(tc + block_size, d_min), d_max + 1):
                pred_profit = power_value * (price_sum[td] - price_sum[tc])
                penalty = power_value * lambda_uncertainty * (
                    uncertainty_sum[td] + uncertainty_sum[tc]
                )
                score = pred_profit - penalty
                if score > best_score:
                    best_score = float(score)
                    best_pred_profit = float(pred_profit)
                    best_tc = tc
                    best_td = td

        power = np.zeros(96, dtype=float)
        traded = bool(best_score > threshold)
        if traded:
            power[best_tc : best_tc + block_size] = -power_value
            power[best_td : best_td + block_size] = power_value
        outputs.append(
            pd.DataFrame(
                {
                    "times": group["times"].to_numpy(),
                    "鐎圭偞妞傛禒閿嬬壐": prices,
                    "power": power,
                }
            )
        )
        meta.append(
            {
                "date": str(date),
                "robust_score": best_score,
                "pred_best_profit": best_pred_profit,
                "lambda_uncertainty": float(lambda_uncertainty),
                "threshold": float(threshold),
                "charge_start": best_tc if traded else None,
                "discharge_start": best_td if traded else None,
                "traded": traded,
            }
        )

    return pd.concat(outputs, ignore_index=True), pd.DataFrame(meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a robust submission using seed prediction uncertainty.")
    parser.add_argument("--price-csv", required=True)
    parser.add_argument("--output", default="outputs/output_robust.csv")
    parser.add_argument("--meta-output", default="outputs/robust_strategy_meta.csv")
    parser.add_argument("--price-col", default="")
    parser.add_argument("--uncertainty-col", default="")
    parser.add_argument("--lambda-uncertainty", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    args = parser.parse_args()

    price_df = pd.read_csv(args.price_csv)
    out, meta = generate_robust_strategy(
        price_df,
        lambda_uncertainty=args.lambda_uncertainty,
        threshold=args.threshold,
        price_col=args.price_col or None,
        uncertainty_col=args.uncertainty_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    if args.meta_output:
        Path(args.meta_output).parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(args.meta_output, index=False)
    print(
        f"saved_robust_submission={args.output}, rows={len(out)}, "
        f"lambda_uncertainty={args.lambda_uncertainty}, threshold={args.threshold}"
    )


if __name__ == "__main__":
    main()
