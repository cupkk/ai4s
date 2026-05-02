from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StrategyResult:
    power: np.ndarray
    best_profit: float
    charge_start: Optional[int]
    discharge_start: Optional[int]
    traded: bool


def optimize_one_day(
    prices: Iterable[float],
    threshold: float = 0.0,
    block_size: int = 8,
    power_value: float = 1000.0,
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
) -> StrategyResult:
    prices_arr = np.asarray(list(prices), dtype=float)
    if prices_arr.shape[0] != 96:
        raise ValueError(f"one day must contain 96 price points, got {prices_arr.shape[0]}")

    power = np.zeros(96, dtype=float)
    block_sum = np.array(
        [prices_arr[i : i + block_size].sum() for i in range(96 - block_size + 1)]
    )

    best_profit = -np.inf
    best_tc: Optional[int] = None
    best_td: Optional[int] = None

    max_charge_start = 96 - 2 * block_size
    max_start = 96 - block_size
    c_min = max(0, int(charge_start_min))
    c_max = min(max_charge_start, int(charge_start_max))
    d_min = max(block_size, int(discharge_start_min))
    d_max = min(max_start, int(discharge_start_max))
    if c_min > c_max:
        raise ValueError(f"invalid charge start bounds: {c_min}>{c_max}")
    if d_min > d_max:
        raise ValueError(f"invalid discharge start bounds: {d_min}>{d_max}")

    for tc in range(c_min, c_max + 1):
        charge_cost = block_sum[tc]
        for td in range(max(tc + block_size, d_min), d_max + 1):
            profit = power_value * (block_sum[td] - charge_cost)
            if profit > best_profit:
                best_profit = float(profit)
                best_tc = tc
                best_td = td

    traded = bool(best_profit > threshold)
    if traded:
        if best_tc is None or best_td is None:
            raise RuntimeError("internal optimizer error: missing trade starts")
        power[best_tc : best_tc + block_size] = -power_value
        power[best_td : best_td + block_size] = power_value

    return StrategyResult(
        power=power,
        best_profit=float(best_profit),
        charge_start=best_tc if traded else None,
        discharge_start=best_td if traded else None,
        traded=traded,
    )


def infer_price_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"price column not found: {preferred}")
        return preferred
    for col in ["实时价格", "瀹炴椂浠锋牸", "pred_price", "prediction", "A"]:
        if col in df.columns:
            return col
    numeric_cols = [
        col
        for col in df.columns
        if col != "times" and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError("cannot infer price column, expected one of: 实时价格, pred_price, prediction, A")


def generate_strategy(
    price_df: pd.DataFrame,
    threshold: float = 0.0,
    price_col: Optional[str] = None,
    strict_96: bool = True,
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
    threshold_by_month: Optional[Dict[int, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "times" not in price_df.columns:
        raise ValueError("missing required column: times")

    price_col = infer_price_column(price_df, price_col)
    df = price_df.copy()
    df["times"] = pd.to_datetime(df["times"])
    df = df.sort_values("times").reset_index(drop=True)
    df["__date__"] = df["times"].dt.date

    outputs = []
    metadata = []
    for date, group in df.groupby("__date__", sort=True):
        group = group.sort_values("times").copy()
        if strict_96 and len(group) != 96:
            raise ValueError(f"{date} must contain 96 rows, got {len(group)}")

        month = int(pd.Timestamp(group["times"].iloc[0]).month)
        day_threshold = (
            float(threshold_by_month[month])
            if threshold_by_month and month in threshold_by_month
            else float(threshold)
        )
        result = optimize_one_day(
            group[price_col].to_numpy(),
            threshold=day_threshold,
            charge_start_min=charge_start_min,
            charge_start_max=charge_start_max,
            discharge_start_min=discharge_start_min,
            discharge_start_max=discharge_start_max,
        )
        day_out = pd.DataFrame(
            {
                "times": group["times"].to_numpy(),
                "实时价格": group[price_col].to_numpy(dtype=float),
                "power": result.power,
            }
        )
        outputs.append(day_out)
        metadata.append(
            {
                "date": str(date),
                "pred_best_profit": result.best_profit,
                "threshold": day_threshold,
                "charge_start": result.charge_start,
                "discharge_start": result.discharge_start,
                "traded": result.traded,
            }
        )

    if not outputs:
        raise ValueError("no daily groups found in price data")

    return pd.concat(outputs, ignore_index=True), pd.DataFrame(metadata)


def save_strategy(
    price_df: pd.DataFrame,
    save_path: str,
    threshold: float = 0.0,
    price_col: Optional[str] = None,
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
) -> pd.DataFrame:
    out, _ = generate_strategy(
        price_df,
        threshold=threshold,
        price_col=price_col,
        charge_start_min=charge_start_min,
        charge_start_max=charge_start_max,
        discharge_start_min=discharge_start_min,
        discharge_start_max=discharge_start_max,
    )
    out.to_csv(save_path, index=False)
    return out


def evaluate_power(prices: Iterable[float], power: Iterable[float]) -> float:
    prices_arr = np.asarray(list(prices), dtype=float)
    power_arr = np.asarray(list(power), dtype=float)
    if prices_arr.shape != power_arr.shape:
        raise ValueError(f"prices and power shape mismatch: {prices_arr.shape} vs {power_arr.shape}")
    return float(np.sum(prices_arr * power_arr))
