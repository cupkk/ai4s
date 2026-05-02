from __future__ import annotations

from typing import Dict, Mapping, Sequence

import pandas as pd

from .features import TARGET_COL, TIME_COL


RECENT_WINDOWS_DAYS = [7, 14, 28]
STATS = ["mean", "median", "std", "p10", "p90"]


def _add_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    dt = out[TIME_COL].dt
    out["month"] = dt.month
    out["day"] = dt.day
    out["slot"] = dt.hour * 4 + dt.minute // 15
    return out


def _key(values: Sequence[object]) -> str:
    return "|".join(str(int(v)) for v in values)


def _series_to_map(series: pd.Series) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in series.items():
        if not isinstance(key, tuple):
            key = (key,)
        if pd.notna(value):
            result[_key(key)] = float(value)
    return result


def _group_stat_map(df: pd.DataFrame, keys: Sequence[str], target_col: str, stat: str) -> Dict[str, float]:
    grouped = df.groupby(list(keys), dropna=False)[target_col]
    if stat == "mean":
        values = grouped.mean()
    elif stat == "median":
        values = grouped.median()
    elif stat == "std":
        values = grouped.std().fillna(0.0)
    elif stat == "p10":
        values = grouped.quantile(0.10)
    elif stat == "p90":
        values = grouped.quantile(0.90)
    else:
        raise ValueError(f"unsupported stat: {stat}")
    return _series_to_map(values)


def fit_price_history_features(
    label_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    recent_windows_days: Sequence[int] = RECENT_WINDOWS_DAYS,
) -> Dict[str, object]:
    if target_col not in label_df.columns:
        raise ValueError(f"missing target column for price history: {target_col}")
    work = _add_keys(label_df[[TIME_COL, target_col]].copy())
    global_stats = {
        "mean": float(work[target_col].mean()),
        "median": float(work[target_col].median()),
        "std": float(work[target_col].std()),
        "p10": float(work[target_col].quantile(0.10)),
        "p90": float(work[target_col].quantile(0.90)),
    }

    stats: Dict[str, object] = {
        "target_col": target_col,
        "global": global_stats,
        "month_day_slot_mean": _group_stat_map(work, ["month", "day", "slot"], target_col, "mean"),
    }
    for stat in STATS:
        stats[f"month_slot_{stat}"] = _group_stat_map(work, ["month", "slot"], target_col, stat)
        stats[f"slot_{stat}"] = _group_stat_map(work, ["slot"], target_col, stat)

    winter = work[work["month"].isin([1, 2, 12])]
    for stat in STATS:
        stats[f"winter_slot_{stat}"] = (
            _group_stat_map(winter, ["slot"], target_col, stat) if not winter.empty else {}
        )

    max_time = work[TIME_COL].max()
    for window_days in recent_windows_days:
        start = max_time - pd.Timedelta(days=int(window_days))
        recent = work[work[TIME_COL] > start]
        for stat in STATS:
            stats[f"recent_{window_days}d_slot_{stat}"] = (
                _group_stat_map(recent, ["slot"], target_col, stat) if not recent.empty else {}
            )
    return stats


def _lookup(
    stats: Mapping[str, object],
    map_name: str,
    keys: Sequence[object],
    default_stat: str = "mean",
) -> float:
    mapping = stats.get(map_name, {})
    default = stats.get("global", {})
    if not isinstance(default, Mapping):
        default = {}
    fallback = float(default.get(default_stat, default.get("mean", 0.0)))
    if not isinstance(mapping, Mapping):
        return fallback
    return float(mapping.get(_key(keys), fallback))


def add_price_history_features(
    feature_df: pd.DataFrame,
    stats: Mapping[str, object] | None,
    recent_windows_days: Sequence[int] = RECENT_WINDOWS_DAYS,
) -> pd.DataFrame:
    if not stats:
        return feature_df.copy()
    out = _add_keys(feature_df)

    out["price_hist_same_month_day_slot"] = [
        _lookup(stats, "month_day_slot_mean", [month, day, slot], "mean")
        for month, day, slot in zip(out["month"], out["day"], out["slot"])
    ]
    for stat in STATS:
        out[f"price_hist_month_slot_{stat}"] = [
            _lookup(stats, f"month_slot_{stat}", [month, slot], stat)
            for month, slot in zip(out["month"], out["slot"])
        ]
        out[f"price_hist_slot_{stat}"] = [
            _lookup(stats, f"slot_{stat}", [slot], stat) for slot in out["slot"]
        ]
        out[f"price_hist_winter_slot_{stat}"] = [
            _lookup(stats, f"winter_slot_{stat}", [slot], stat) for slot in out["slot"]
        ]

    for window_days in recent_windows_days:
        for stat in STATS:
            out[f"price_hist_recent_{window_days}d_slot_{stat}"] = [
                _lookup(stats, f"recent_{window_days}d_slot_{stat}", [slot], stat)
                for slot in out["slot"]
            ]
    return out
