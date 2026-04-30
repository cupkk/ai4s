from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TIME_COL = "times"
TARGET_COL = "A"

DEFAULT_BASE_FEATURES = [
    "系统负荷预测值",
    "风光总加预测值",
    "联络线预测值",
    "风电预测值",
    "光伏预测值",
    "水电预测值",
    "非市场化机组预测值",
]

EXTRA_FEATURE_PREFIXES = ("nwp_", "price_lag_", "price_roll_")

TIME_FEATURES = [
    "hour",
    "minute",
    "dayofweek",
    "month",
    "day",
    "dayofyear",
    "weekofyear",
    "quarter",
    "is_weekend",
    "slot",
    "slot_sin",
    "slot_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

HISTORY_FEATURES = [
    "hist_slot_mean",
    "hist_slot_std",
    "hist_slot_p10",
    "hist_slot_p90",
    "hist_month_slot_mean",
    "hist_month_slot_std",
    "hist_month_slot_p10",
    "hist_month_slot_p90",
    "hist_dow_slot_mean",
    "hist_dow_slot_std",
    "hist_dow_slot_p10",
    "hist_dow_slot_p90",
]

EXACT_CALENDAR_HISTORY_FEATURES = [
    "hist_dayofyear_slot_mean",
    "hist_month_day_slot_mean",
]

EPS = 1e-6

FORECAST_VALUE_PAIRS = [
    ("系统负荷实际值", "系统负荷预测值"),
    ("风光总加实际值", "风光总加预测值"),
    ("联络线实际值", "联络线预测值"),
    ("风电实际值", "风电预测值"),
    ("光伏实际值", "光伏预测值"),
    ("水电实际值", "水电预测值"),
    ("非市场化机组实际值", "非市场化机组预测值"),
]


@dataclass
class FeatureBuildResult:
    frame: pd.DataFrame
    feature_columns: List[str]


def ensure_datetime(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"missing required time column: {time_col}")
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    return out


def add_time_features(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    out = ensure_datetime(df, time_col=time_col)
    dt = out[time_col].dt
    slot = dt.hour * 4 + (dt.minute // 15)

    out["hour"] = dt.hour
    out["minute"] = dt.minute
    out["dayofweek"] = dt.dayofweek
    out["month"] = dt.month
    out["day"] = dt.day
    out["dayofyear"] = dt.dayofyear
    out["weekofyear"] = dt.isocalendar().week.astype(int)
    out["quarter"] = dt.quarter
    out["is_weekend"] = (dt.dayofweek >= 5).astype(int)
    out["slot"] = slot.astype(int)

    out["slot_sin"] = np.sin(2 * np.pi * out["slot"] / 96.0)
    out["slot_cos"] = np.cos(2 * np.pi * out["slot"] / 96.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def _has_columns(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(col in df.columns for col in cols)


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / (denom.abs() + EPS)


def add_balance_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    made: List[str] = []

    load = "系统负荷预测值"
    fg = "风光总加预测值"
    intertie = "联络线预测值"
    wind = "风电预测值"
    pv = "光伏预测值"
    hydro = "水电预测值"
    non_market = "非市场化机组预测值"

    if _has_columns(out, [load, wind, pv, hydro, non_market]):
        renewable = out[wind] + out[pv]
        fixed_supply = renewable + out[hydro] + out[non_market]
        out["renewable_total"] = renewable
        out["net_load"] = out[load] - fixed_supply
        out["renewable_ratio"] = _safe_ratio(renewable, out[load])
        out["wind_ratio"] = _safe_ratio(out[wind], out[load])
        out["pv_ratio"] = _safe_ratio(out[pv], out[load])
        out["hydro_ratio"] = _safe_ratio(out[hydro], out[load])
        out["non_market_ratio"] = _safe_ratio(out[non_market], out[load])
        out["fixed_supply"] = fixed_supply
        out["supply_margin"] = fixed_supply - out[load]
        out["demand_supply_ratio"] = _safe_ratio(out[load], fixed_supply)
        made.extend(
            [
                "renewable_total",
                "net_load",
                "renewable_ratio",
                "wind_ratio",
                "pv_ratio",
                "hydro_ratio",
                "non_market_ratio",
                "fixed_supply",
                "supply_margin",
                "demand_supply_ratio",
            ]
        )

    if _has_columns(out, [load, fg]):
        out["load_minus_windsolar_total"] = out[load] - out[fg]
        out["windsolar_ratio"] = _safe_ratio(out[fg], out[load])
        made.extend(["load_minus_windsolar_total", "windsolar_ratio"])

    if _has_columns(out, [fg, wind, pv]):
        out["windsolar_residual"] = out[fg] - out[wind] - out[pv]
        made.append("windsolar_residual")

    if _has_columns(out, [load, wind, pv, hydro, non_market, intertie]):
        out["net_load_with_intertie"] = (
            out[load] - out[wind] - out[pv] - out[hydro] - out[non_market] - out[intertie]
        )
        out["net_load_intertie_minus"] = out["net_load_with_intertie"]
        out["net_load_intertie_plus"] = (
            out[load] - out[wind] - out[pv] - out[hydro] - out[non_market] + out[intertie]
        )
        made.extend(["net_load_with_intertie", "net_load_intertie_minus", "net_load_intertie_plus"])

    return out, made


def add_ramp_features(
    df: pd.DataFrame,
    source_cols: Iterable[str],
    time_col: str = TIME_COL,
) -> Tuple[pd.DataFrame, List[str]]:
    out = ensure_datetime(df, time_col=time_col)
    source_cols = [col for col in source_cols if col in out.columns]
    if not source_cols:
        return out, []

    order_col = "__row_order__"
    out[order_col] = np.arange(len(out))
    out["__date__"] = out[time_col].dt.date
    sorted_out = out.sort_values([time_col, order_col]).copy()

    made: List[str] = []
    for col in source_cols:
        diff1 = f"{col}_diff1"
        diff4 = f"{col}_diff4"
        roll4 = f"{col}_roll4_mean"
        roll8 = f"{col}_roll8_mean"

        grouped = sorted_out.groupby("__date__", sort=False)[col]
        sorted_out[diff1] = grouped.diff(1).fillna(0.0)
        sorted_out[diff4] = grouped.diff(4).fillna(0.0)
        sorted_out[roll4] = grouped.transform(lambda s: s.rolling(4, min_periods=1).mean())
        sorted_out[roll8] = grouped.transform(lambda s: s.rolling(8, min_periods=1).mean())
        made.extend([diff1, diff4, roll4, roll8])

    sorted_out = sorted_out.sort_values(order_col).drop(columns=[order_col, "__date__"])
    return sorted_out, made


def _group_mean_map(df: pd.DataFrame, keys: Sequence[str], target_col: str) -> Dict[str, float]:
    grouped = df.groupby(list(keys), dropna=False)[target_col].mean()
    result: Dict[str, float] = {}
    for key, value in grouped.items():
        if not isinstance(key, tuple):
            key = (key,)
        result["|".join(str(int(k)) for k in key)] = float(value)
    return result


def _group_stat_map(
    df: pd.DataFrame,
    keys: Sequence[str],
    target_col: str,
    stat: str,
) -> Dict[str, float]:
    grouped = df.groupby(list(keys), dropna=False)[target_col]
    if stat == "std":
        values = grouped.std().fillna(0.0)
    elif stat == "p10":
        values = grouped.quantile(0.10)
    elif stat == "p90":
        values = grouped.quantile(0.90)
    else:
        raise ValueError(f"unsupported group stat: {stat}")

    result: Dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, tuple):
            key = (key,)
        result["|".join(str(int(k)) for k in key)] = float(value)
    return result


def fit_history_stats(df: pd.DataFrame, target_col: str = TARGET_COL) -> Dict[str, object]:
    if target_col not in df.columns:
        raise ValueError(f"cannot fit history stats without target column: {target_col}")

    feat = add_time_features(df)
    global_mean = float(feat[target_col].mean())
    global_std = float(feat[target_col].std())
    global_p10 = float(feat[target_col].quantile(0.10))
    global_p90 = float(feat[target_col].quantile(0.90))
    return {
        "target_col": target_col,
        "global_mean": global_mean,
        "global_std": global_std,
        "global_p10": global_p10,
        "global_p90": global_p90,
        "slot_mean": _group_mean_map(feat, ["slot"], target_col),
        "slot_std": _group_stat_map(feat, ["slot"], target_col, "std"),
        "slot_p10": _group_stat_map(feat, ["slot"], target_col, "p10"),
        "slot_p90": _group_stat_map(feat, ["slot"], target_col, "p90"),
        "month_slot_mean": _group_mean_map(feat, ["month", "slot"], target_col),
        "month_slot_std": _group_stat_map(feat, ["month", "slot"], target_col, "std"),
        "month_slot_p10": _group_stat_map(feat, ["month", "slot"], target_col, "p10"),
        "month_slot_p90": _group_stat_map(feat, ["month", "slot"], target_col, "p90"),
        "dow_slot_mean": _group_mean_map(feat, ["dayofweek", "slot"], target_col),
        "dow_slot_std": _group_stat_map(feat, ["dayofweek", "slot"], target_col, "std"),
        "dow_slot_p10": _group_stat_map(feat, ["dayofweek", "slot"], target_col, "p10"),
        "dow_slot_p90": _group_stat_map(feat, ["dayofweek", "slot"], target_col, "p90"),
        "dayofyear_slot_mean": _group_mean_map(feat, ["dayofyear", "slot"], target_col),
        "month_day_slot_mean": _group_mean_map(feat, ["month", "day", "slot"], target_col),
        "forecast_bias": fit_forecast_bias_stats(feat),
    }


def fit_forecast_bias_stats(df: pd.DataFrame) -> Dict[str, object]:
    stats: Dict[str, object] = {}
    for actual_col, pred_col in FORECAST_VALUE_PAIRS:
        if actual_col not in df.columns or pred_col not in df.columns:
            continue
        bias_col = f"__bias__{pred_col}"
        work = df.copy()
        work[bias_col] = work[actual_col] - work[pred_col]
        stats[pred_col] = {
            "global_mean": float(work[bias_col].mean()),
            "slot_mean": _group_mean_map(work, ["slot"], bias_col),
            "month_slot_mean": _group_mean_map(work, ["month", "slot"], bias_col),
            "dow_slot_mean": _group_mean_map(work, ["dayofweek", "slot"], bias_col),
        }
    return stats


def _lookup(stats: Mapping[str, object], map_name: str, keys: Sequence[object]) -> float:
    mapping = stats.get(map_name, {})
    if not isinstance(mapping, Mapping):
        return float(stats.get("global_mean", 0.0))
    key = "|".join(str(int(k)) for k in keys)
    return float(mapping.get(key, stats.get("global_mean", 0.0)))


def _lookup_stat(
    stats: Mapping[str, object],
    map_name: str,
    keys: Sequence[object],
    default_name: str,
) -> float:
    mapping = stats.get(map_name, {})
    if not isinstance(mapping, Mapping):
        return float(stats.get(default_name, 0.0))
    key = "|".join(str(int(k)) for k in keys)
    return float(mapping.get(key, stats.get(default_name, 0.0)))


def add_history_features(
    df: pd.DataFrame,
    stats: Optional[Mapping[str, object]],
    use_exact_calendar_history: bool = False,
    use_forecast_bias: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    if not stats:
        return out, []

    if not _has_columns(out, ["slot", "month", "dayofweek"]):
        out = add_time_features(out)

    out["hist_slot_mean"] = [
        _lookup(stats, "slot_mean", [slot]) for slot in out["slot"].to_numpy()
    ]
    out["hist_slot_std"] = [
        _lookup_stat(stats, "slot_std", [slot], "global_std") for slot in out["slot"].to_numpy()
    ]
    out["hist_slot_p10"] = [
        _lookup_stat(stats, "slot_p10", [slot], "global_p10") for slot in out["slot"].to_numpy()
    ]
    out["hist_slot_p90"] = [
        _lookup_stat(stats, "slot_p90", [slot], "global_p90") for slot in out["slot"].to_numpy()
    ]
    out["hist_month_slot_mean"] = [
        _lookup(stats, "month_slot_mean", [month, slot])
        for month, slot in zip(out["month"].to_numpy(), out["slot"].to_numpy())
    ]
    out["hist_month_slot_std"] = [
        _lookup_stat(stats, "month_slot_std", [month, slot], "global_std")
        for month, slot in zip(out["month"].to_numpy(), out["slot"].to_numpy())
    ]
    out["hist_month_slot_p10"] = [
        _lookup_stat(stats, "month_slot_p10", [month, slot], "global_p10")
        for month, slot in zip(out["month"].to_numpy(), out["slot"].to_numpy())
    ]
    out["hist_month_slot_p90"] = [
        _lookup_stat(stats, "month_slot_p90", [month, slot], "global_p90")
        for month, slot in zip(out["month"].to_numpy(), out["slot"].to_numpy())
    ]
    out["hist_dow_slot_mean"] = [
        _lookup(stats, "dow_slot_mean", [dow, slot])
        for dow, slot in zip(out["dayofweek"].to_numpy(), out["slot"].to_numpy())
    ]
    out["hist_dow_slot_std"] = [
        _lookup_stat(stats, "dow_slot_std", [dow, slot], "global_std")
        for dow, slot in zip(out["dayofweek"].to_numpy(), out["slot"].to_numpy())
    ]
    out["hist_dow_slot_p10"] = [
        _lookup_stat(stats, "dow_slot_p10", [dow, slot], "global_p10")
        for dow, slot in zip(out["dayofweek"].to_numpy(), out["slot"].to_numpy())
    ]
    out["hist_dow_slot_p90"] = [
        _lookup_stat(stats, "dow_slot_p90", [dow, slot], "global_p90")
        for dow, slot in zip(out["dayofweek"].to_numpy(), out["slot"].to_numpy())
    ]
    made = HISTORY_FEATURES.copy()

    if use_exact_calendar_history:
        out["hist_dayofyear_slot_mean"] = [
            _lookup(stats, "dayofyear_slot_mean", [dayofyear, slot])
            for dayofyear, slot in zip(out["dayofyear"].to_numpy(), out["slot"].to_numpy())
        ]
        out["hist_month_day_slot_mean"] = [
            _lookup(stats, "month_day_slot_mean", [month, day, slot])
            for month, day, slot in zip(
                out["month"].to_numpy(), out["day"].to_numpy(), out["slot"].to_numpy()
            )
        ]
        made.extend(EXACT_CALENDAR_HISTORY_FEATURES)

    if use_forecast_bias:
        out, bias_cols = add_forecast_bias_features(out, stats)
        made.extend(bias_cols)

    return out, made


def add_forecast_bias_features(
    df: pd.DataFrame,
    stats: Mapping[str, object],
) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    bias_stats = stats.get("forecast_bias", {})
    if not isinstance(bias_stats, Mapping):
        return out, []

    made: List[str] = []
    for _, pred_col in FORECAST_VALUE_PAIRS:
        if pred_col not in out.columns or pred_col not in bias_stats:
            continue
        per_col = bias_stats[pred_col]
        if not isinstance(per_col, Mapping):
            continue

        slot_col = f"{pred_col}_bias_slot_mean"
        month_slot_col = f"{pred_col}_bias_month_slot_mean"
        dow_slot_col = f"{pred_col}_bias_dow_slot_mean"
        corrected_slot_col = f"{pred_col}_corrected_slot"
        corrected_month_slot_col = f"{pred_col}_corrected_month_slot"

        out[slot_col] = [_lookup(per_col, "slot_mean", [slot]) for slot in out["slot"].to_numpy()]
        out[month_slot_col] = [
            _lookup(per_col, "month_slot_mean", [month, slot])
            for month, slot in zip(out["month"].to_numpy(), out["slot"].to_numpy())
        ]
        out[dow_slot_col] = [
            _lookup(per_col, "dow_slot_mean", [dow, slot])
            for dow, slot in zip(out["dayofweek"].to_numpy(), out["slot"].to_numpy())
        ]
        out[corrected_slot_col] = out[pred_col] + out[slot_col]
        out[corrected_month_slot_col] = out[pred_col] + out[month_slot_col]
        made.extend(
            [
                slot_col,
                month_slot_col,
                dow_slot_col,
                corrected_slot_col,
                corrected_month_slot_col,
            ]
        )
    return out, made


def build_features(
    df: pd.DataFrame,
    history_stats: Optional[Mapping[str, object]] = None,
    base_features: Sequence[str] = DEFAULT_BASE_FEATURES,
    use_exact_calendar_history: bool = False,
    use_forecast_bias: bool = False,
) -> FeatureBuildResult:
    out = add_time_features(df)
    feature_columns: List[str] = [col for col in base_features if col in out.columns]
    feature_columns.extend(
        [
            col
            for col in out.columns
            if col.startswith(EXTRA_FEATURE_PREFIXES)
            and pd.api.types.is_numeric_dtype(out[col])
        ]
    )
    feature_columns.extend([col for col in TIME_FEATURES if col in out.columns])

    out, balance_cols = add_balance_features(out)
    feature_columns.extend(balance_cols)

    ramp_sources = [col for col in base_features if col in out.columns]
    ramp_sources.extend(balance_cols)
    out, ramp_cols = add_ramp_features(out, ramp_sources)
    feature_columns.extend(ramp_cols)

    out, history_cols = add_history_features(
        out,
        history_stats,
        use_exact_calendar_history=use_exact_calendar_history,
        use_forecast_bias=use_forecast_bias,
    )
    feature_columns.extend(history_cols)

    seen = set()
    unique_features: List[str] = []
    for col in feature_columns:
        if col not in seen and col in out.columns:
            seen.add(col)
            unique_features.append(col)

    return FeatureBuildResult(frame=out, feature_columns=unique_features)


def align_feature_frame(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_columns:
        if col not in out.columns:
            out[col] = 0.0
    return out.loc[:, list(feature_columns)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
