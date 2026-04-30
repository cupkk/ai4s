from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .features import DEFAULT_BASE_FEATURES, TARGET_COL, TIME_COL
from .train_lgb import load_training_frame, rmse


def _linear_rmse(train_x: pd.Series, train_y: pd.Series, val_x: pd.Series, val_y: pd.Series) -> float:
    x = train_x.to_numpy(dtype=float)
    y = train_y.to_numpy(dtype=float)
    if np.std(x) <= 1e-12:
        return float("nan")
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * val_x.to_numpy(dtype=float) + intercept
    return rmse(val_y.to_numpy(dtype=float), pred)


def diagnose(df: pd.DataFrame, val_days: int) -> pd.DataFrame:
    load, _, intertie, wind, pv, hydro, non_market = DEFAULT_BASE_FEATURES
    required = [load, intertie, wind, pv, hydro, non_market, TARGET_COL, TIME_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    work = df.copy()
    work[TIME_COL] = pd.to_datetime(work[TIME_COL])
    fixed_supply = work[wind] + work[pv] + work[hydro] + work[non_market]
    work["net_load_minus_intertie"] = work[load] - fixed_supply - work[intertie]
    work["net_load_plus_intertie"] = work[load] - fixed_supply + work[intertie]
    work["intertie_raw"] = work[intertie]

    dates = sorted(work[TIME_COL].dt.date.unique())
    n_val = min(max(1, val_days), len(dates) - 1)
    val_dates = set(dates[-n_val:])
    train = work[~work[TIME_COL].dt.date.isin(val_dates)]
    val = work[work[TIME_COL].dt.date.isin(val_dates)]

    rows: List[Dict[str, object]] = []
    for col in ["intertie_raw", "net_load_minus_intertie", "net_load_plus_intertie"]:
        rows.append(
            {
                "feature": col,
                "pearson_corr_with_price": float(work[col].corr(work[TARGET_COL], method="pearson")),
                "spearman_corr_with_price": float(work[col].corr(work[TARGET_COL], method="spearman")),
                "linear_val_rmse": _linear_rmse(train[col], train[TARGET_COL], val[col], val[TARGET_COL]),
                "val_days": n_val,
            }
        )
    return pd.DataFrame(rows).sort_values("linear_val_rmse", ascending=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether intertie should be added or subtracted.")
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--val-days", type=int, default=60)
    parser.add_argument("--output", default="outputs/intertie_sign_diagnostic.csv")
    args = parser.parse_args()

    df = load_training_frame(args.train_feature, args.train_label, target_col=TARGET_COL)
    result = diagnose(df, val_days=args.val_days)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))
    print(f"saved_intertie_diagnostic={output_path}")


if __name__ == "__main__":
    main()
