from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import List

import pandas as pd

from .validate_profit import backtest_predictions, parse_threshold_grid


def _parse_months(text: str) -> List[int]:
    months = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not months:
        raise ValueError("at least one month is required")
    for month in months:
        if month < 1 or month > 12:
            raise ValueError(f"invalid month: {month}")
    return months


def tune_monthly_thresholds(
    df: pd.DataFrame,
    months: List[int],
    thresholds: List[float],
    pred_col: str,
    true_col: str,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
) -> pd.DataFrame:
    rows = []
    for combo in itertools.product(thresholds, repeat=len(months)):
        threshold_by_month = {month: float(value) for month, value in zip(months, combo)}
        summary, _ = backtest_predictions(
            df,
            threshold=0.0,
            pred_col=pred_col,
            true_col=true_col,
            charge_start_min=charge_start_min,
            charge_start_max=charge_start_max,
            discharge_start_min=discharge_start_min,
            discharge_start_max=discharge_start_max,
            threshold_by_month=threshold_by_month,
        )
        rows.append({**summary, "threshold_by_month": json.dumps(threshold_by_month, sort_keys=True)})
    return pd.DataFrame(rows).sort_values(
        ["avg_profit", "capture_ratio", "loss_days"],
        ascending=[False, False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune per-month trade thresholds on validation predictions.")
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--pred-col", default="pred_price")
    parser.add_argument("--true-col", default="A")
    parser.add_argument("--months", default="1,2")
    parser.add_argument("--threshold-grid", default="0,500,1000,2000,3000,5000,8000,10000")
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--output", default="outputs/monthly_threshold_search.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    result = tune_monthly_thresholds(
        df,
        months=_parse_months(args.months),
        thresholds=parse_threshold_grid(args.threshold_grid),
        pred_col=args.pred_col,
        true_col=args.true_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.head(20).to_string(index=False))
    print(f"saved_monthly_threshold_search={output_path}")


if __name__ == "__main__":
    main()
