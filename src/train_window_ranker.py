from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, align_feature_frame, build_features, fit_history_stats
from .nwp_features import load_or_build_nwp_features, merge_nwp_features
from .price_history_features import add_price_history_features, fit_price_history_features
from .storage_optimizer import evaluate_power, optimize_one_day
from .train_lgb import (
    DEFAULT_PARAMS,
    load_training_frame,
    params_for_seed,
    parse_seeds,
    split_by_day,
    train_booster,
)


DEFAULT_POINT_FEATURES = [
    "price_hist_same_month_day_slot",
    "price_hist_month_slot_median",
    "price_hist_month_slot_p10",
    "price_hist_month_slot_p90",
    "price_hist_recent_28d_slot_mean",
    "price_hist_recent_28d_slot_std",
    "hist_slot_mean",
    "hist_month_slot_mean",
    "net_load",
    "renewable_ratio",
    "supply_margin",
    "net_load_intertie_minus",
    "net_load_intertie_plus",
    "nwp_ghi_mean",
    "nwp_wind_speed_mean",
    "nwp_wind_speed_cube_mean",
    "nwp_t2m_mean",
    "nwp_tcc_mean",
]

LABEL_COLUMNS = {
    "true_window_profit",
    "true_delta_profit",
    "baseline_true_window_profit",
}
DEFAULT_BASELINE_WINDOW_SCORE_COL = "spread_price_hist_recent_28d_slot_mean"


def _block_values(values: np.ndarray, block_size: int, reducer: str) -> np.ndarray:
    out = []
    for start in range(len(values) - block_size + 1):
        window = values[start : start + block_size]
        if reducer == "sum":
            out.append(float(np.sum(window)))
        elif reducer == "mean":
            out.append(float(np.mean(window)))
        elif reducer == "std":
            out.append(float(np.std(window)))
        elif reducer == "min":
            out.append(float(np.min(window)))
        elif reducer == "max":
            out.append(float(np.max(window)))
        else:
            raise ValueError(f"unsupported reducer: {reducer}")
    return np.asarray(out, dtype=float)


def _candidate_frame_for_day(
    day_features: pd.DataFrame,
    true_prices: np.ndarray | None,
    point_features: Sequence[str],
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
    block_size: int = 8,
    power_value: float = 1000.0,
) -> pd.DataFrame:
    if len(day_features) != 96:
        raise ValueError(f"expected 96 rows per day, got {len(day_features)}")
    max_charge_start = 96 - 2 * block_size
    max_start = 96 - block_size
    c_min = max(0, int(charge_start_min))
    c_max = min(max_charge_start, int(charge_start_max))
    d_min = max(block_size, int(discharge_start_min))
    d_max = min(max_start, int(discharge_start_max))

    block_means: Dict[str, np.ndarray] = {}
    shape_reducers = ("std", "min", "max")
    block_shape_values: Dict[tuple[str, str], np.ndarray] = {}
    for col in point_features:
        values = day_features[col].to_numpy(dtype=float)
        block_means[col] = _block_values(values, block_size, "mean")
        for reducer in shape_reducers:
            block_shape_values[(col, reducer)] = _block_values(values, block_size, reducer)
    if true_prices is not None:
        true_block_sum = _block_values(true_prices, block_size, "sum")

    rows = []
    for tc in range(c_min, c_max + 1):
        for td in range(max(tc + block_size, d_min), d_max + 1):
            row = {
                "charge_start": tc,
                "discharge_start": td,
                "gap_slots": td - tc,
                "charge_hour": tc / 4.0,
                "discharge_hour": td / 4.0,
            }
            for col, values in block_means.items():
                charge_value = values[tc]
                discharge_value = values[td]
                row[f"charge_{col}"] = charge_value
                row[f"discharge_{col}"] = discharge_value
                row[f"spread_{col}"] = discharge_value - charge_value
                for reducer in shape_reducers:
                    shape_values = block_shape_values[(col, reducer)]
                    charge_shape_value = shape_values[tc]
                    discharge_shape_value = shape_values[td]
                    row[f"charge_{col}_{reducer}"] = charge_shape_value
                    row[f"discharge_{col}_{reducer}"] = discharge_shape_value
                    row[f"spread_{col}_{reducer}"] = discharge_shape_value - charge_shape_value
            if true_prices is not None:
                row["true_window_profit"] = power_value * (true_block_sum[td] - true_block_sum[tc])
            rows.append(row)
    return pd.DataFrame(rows)


def build_window_dataset(
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    point_features: Sequence[str],
    target_col: str,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
    include_target: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df[[TIME_COL] + ([target_col] if include_target else [])].copy()
    work[TIME_COL] = pd.to_datetime(work[TIME_COL])
    features = feature_frame.copy()
    features[TIME_COL] = pd.to_datetime(df[TIME_COL]).to_numpy()
    features["__date__"] = features[TIME_COL].dt.date
    work["__date__"] = work[TIME_COL].dt.date

    candidates = []
    meta_rows = []
    for date, day_idx in features.groupby("__date__", sort=True).groups.items():
        day_features = features.loc[day_idx, list(point_features)].reset_index(drop=True)
        if len(day_features) != 96:
            continue
        true_prices = None
        if include_target:
            true_prices = work.loc[day_idx, target_col].to_numpy(dtype=float)
        day_candidates = _candidate_frame_for_day(
            day_features,
            true_prices=true_prices,
            point_features=point_features,
            charge_start_min=charge_start_min,
            charge_start_max=charge_start_max,
            discharge_start_min=discharge_start_min,
            discharge_start_max=discharge_start_max,
        )
        day_candidates["date"] = str(date)
        candidates.append(day_candidates)
        meta_rows.append({"date": str(date), "candidate_count": len(day_candidates)})
    if not candidates:
        raise ValueError("no complete daily windows available")
    return pd.concat(candidates, ignore_index=True), pd.DataFrame(meta_rows)


def baseline_windows_from_submission(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "times" not in df.columns or "power" not in df.columns:
        raise ValueError(f"{path} must contain times and power columns")
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)

    rows = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        power = group["power"].to_numpy(dtype=float)
        charge = np.flatnonzero(power < 0)
        discharge = np.flatnonzero(power > 0)
        if len(charge) == 0 or len(discharge) == 0:
            continue
        rows.append(
            {
                "date": date,
                "baseline_charge_start": int(charge[0]),
                "baseline_discharge_start": int(discharge[0]),
            }
        )
    return pd.DataFrame(rows)


def baseline_windows_from_meta(path: str | Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    required = {"date", "charge_start", "discharge_start"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"{path} missing required baseline columns: {sorted(missing)}")
    out = meta.loc[:, ["date", "charge_start", "discharge_start"]].copy()
    out["date"] = out["date"].astype(str)
    out = out.rename(
        columns={
            "charge_start": "baseline_charge_start",
            "discharge_start": "baseline_discharge_start",
        }
    )
    out = out.dropna(subset=["baseline_charge_start", "baseline_discharge_start"])
    out["baseline_charge_start"] = out["baseline_charge_start"].astype(int)
    out["baseline_discharge_start"] = out["baseline_discharge_start"].astype(int)
    return out


def baseline_windows_from_score(candidate_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if score_col not in candidate_df.columns:
        raise ValueError(f"baseline score column not found in window candidates: {score_col}")
    idx = candidate_df.groupby("date", sort=True)[score_col].idxmax()
    selected = candidate_df.loc[idx, ["date", "charge_start", "discharge_start"]].copy()
    selected = selected.rename(
        columns={
            "charge_start": "baseline_charge_start",
            "discharge_start": "baseline_discharge_start",
        }
    )
    selected["baseline_charge_start"] = selected["baseline_charge_start"].astype(int)
    selected["baseline_discharge_start"] = selected["baseline_discharge_start"].astype(int)
    return selected.sort_values("date").reset_index(drop=True)


def prepare_baseline_windows(
    candidate_df: pd.DataFrame,
    score_col: str,
    meta_path: str = "",
    submission_path: str = "",
) -> pd.DataFrame:
    if submission_path:
        baseline = baseline_windows_from_submission(submission_path)
    elif meta_path:
        baseline = baseline_windows_from_meta(meta_path)
    else:
        baseline = baseline_windows_from_score(candidate_df, score_col)
    baseline_dates = set(baseline["date"].astype(str))
    candidate_dates = set(candidate_df["date"].astype(str))
    missing = sorted(candidate_dates.difference(baseline_dates))
    if missing:
        raise ValueError(f"baseline windows missing dates: {missing[:10]}")
    return baseline


def attach_baseline_window_context(
    candidate_df: pd.DataFrame,
    baseline_windows: pd.DataFrame,
) -> pd.DataFrame:
    base = baseline_windows.copy()
    base["date"] = base["date"].astype(str)
    base["baseline_charge_start"] = base["baseline_charge_start"].astype(int)
    base["baseline_discharge_start"] = base["baseline_discharge_start"].astype(int)

    baseline_records = []
    baseline_context_cols = [
        col
        for col in candidate_df.columns
        if col.startswith(("spread_", "charge_", "discharge_"))
        and col not in {"charge_start", "discharge_start"}
    ]
    spread_cols = [col for col in baseline_context_cols if col.startswith("spread_")]
    for _, row in base.iterrows():
        date = row["date"]
        charge_start = int(row["baseline_charge_start"])
        discharge_start = int(row["baseline_discharge_start"])
        match = candidate_df.loc[
            (candidate_df["date"].astype(str) == date)
            & (candidate_df["charge_start"].astype(int) == charge_start)
            & (candidate_df["discharge_start"].astype(int) == discharge_start)
        ]
        if match.empty:
            raise ValueError(
                f"baseline window is outside candidate grid for {date}: "
                f"charge={charge_start}, discharge={discharge_start}"
            )
        baseline_row = match.iloc[0]
        record = {
            "date": date,
            "baseline_charge_start": charge_start,
            "baseline_discharge_start": discharge_start,
            "baseline_gap_slots": int(baseline_row["gap_slots"]),
            "baseline_charge_hour": float(baseline_row["charge_hour"]),
            "baseline_discharge_hour": float(baseline_row["discharge_hour"]),
        }
        if "true_window_profit" in baseline_row.index:
            record["baseline_true_window_profit"] = float(baseline_row["true_window_profit"])
        for col in baseline_context_cols:
            record[f"baseline_{col}"] = float(baseline_row[col])
        baseline_records.append(record)

    context = pd.DataFrame(baseline_records).set_index("date")
    out = candidate_df.copy(deep=False)
    date_key = out["date"].astype(str)
    for col in context.columns:
        out[col] = date_key.map(context[col])
    if out["baseline_charge_start"].isna().any() or out["baseline_discharge_start"].isna().any():
        missing = sorted(set(date_key[out["baseline_charge_start"].isna()]))
        raise ValueError(f"baseline context merge failed for dates: {missing[:10]}")
    out["baseline_charge_start"] = out["baseline_charge_start"].astype(int)
    out["baseline_discharge_start"] = out["baseline_discharge_start"].astype(int)
    out["delta_charge_start"] = out["charge_start"].astype(int) - out["baseline_charge_start"].astype(int)
    out["delta_discharge_start"] = (
        out["discharge_start"].astype(int) - out["baseline_discharge_start"].astype(int)
    )
    out["abs_delta_charge_start"] = out["delta_charge_start"].abs()
    out["abs_delta_discharge_start"] = out["delta_discharge_start"].abs()
    out["delta_gap_slots"] = out["gap_slots"].astype(float) - out["baseline_gap_slots"].astype(float)
    out["same_as_baseline_window"] = (
        (out["delta_charge_start"] == 0) & (out["delta_discharge_start"] == 0)
    ).astype(int)
    for col in spread_cols:
        baseline_col = f"baseline_{col}"
        if baseline_col in out.columns:
            out[f"delta_vs_baseline_{col}"] = out[col].astype(float) - out[baseline_col].astype(float)
    if "true_window_profit" in out.columns and "baseline_true_window_profit" in out.columns:
        out["true_delta_profit"] = (
            out["true_window_profit"].astype(float)
            - out["baseline_true_window_profit"].astype(float)
        )
    return out


def filter_windows_near_baseline_starts(
    candidate_df: pd.DataFrame,
    baseline_windows: pd.DataFrame,
    max_abs_delta: int,
) -> pd.DataFrame:
    """Keep only windows close enough to each date's baseline starts.

    This is used before attaching the wide baseline context columns. The
    replacement-classifier path only ever considers near-baseline moves, so
    filtering first avoids copying a very large all-window table.
    """
    if int(max_abs_delta) < 0:
        return candidate_df
    required = {"date", "charge_start", "discharge_start"}
    missing = required.difference(candidate_df.columns)
    if missing:
        raise ValueError(f"candidate windows missing columns for near-baseline filter: {sorted(missing)}")
    baseline_required = {"date", "baseline_charge_start", "baseline_discharge_start"}
    baseline_missing = baseline_required.difference(baseline_windows.columns)
    if baseline_missing:
        raise ValueError(f"baseline windows missing columns for near-baseline filter: {sorted(baseline_missing)}")

    baseline = baseline_windows.loc[
        :, ["date", "baseline_charge_start", "baseline_discharge_start"]
    ].copy()
    baseline["date"] = baseline["date"].astype(str)
    baseline = baseline.drop_duplicates(subset=["date"]).set_index("date")
    date_key = candidate_df["date"].astype(str)
    base_charge = date_key.map(baseline["baseline_charge_start"])
    base_discharge = date_key.map(baseline["baseline_discharge_start"])
    if base_charge.isna().any() or base_discharge.isna().any():
        missing_dates = sorted(set(date_key[base_charge.isna() | base_discharge.isna()]))
        raise ValueError(f"baseline windows missing dates for near-baseline filter: {missing_dates[:10]}")

    near = (
        (candidate_df["charge_start"].astype(int) - base_charge.astype(int)).abs()
        <= int(max_abs_delta)
    ) & (
        (candidate_df["discharge_start"].astype(int) - base_discharge.astype(int)).abs()
        <= int(max_abs_delta)
    )
    out = candidate_df.loc[near].copy()
    if out.empty:
        raise ValueError(f"near-baseline prefilter removed all windows: max_abs_delta={max_abs_delta}")
    return out.reset_index(drop=True)


def choose_daily_windows(candidate_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    idx = candidate_df.groupby("date", sort=True)[score_col].idxmax()
    return candidate_df.loc[idx].sort_values("date").reset_index(drop=True)


def _group_sizes_by_date(candidate_df: pd.DataFrame) -> list[int]:
    return [int(size) for size in candidate_df.groupby("date", sort=True).size().tolist()]


def _ranker_labels(candidate_df: pd.DataFrame, target_col: str = "true_window_profit") -> np.ndarray:
    """Convert per-day window profits into bounded integer relevance labels."""
    labels = pd.Series(index=candidate_df.index, dtype=float)
    for _, group in candidate_df.groupby("date", sort=True):
        if len(group) <= 1:
            labels.loc[group.index] = 0
            continue
        pct = group[target_col].rank(method="first", pct=True)
        labels.loc[group.index] = np.floor((pct.to_numpy(dtype=float) - 1.0 / len(group)) * 30.0)
    return labels.fillna(0).clip(lower=0, upper=30).astype(int).to_numpy()


def _train_window_booster(
    train_x: pd.DataFrame,
    train_profit: np.ndarray,
    val_x: pd.DataFrame | None,
    val_profit: np.ndarray | None,
    params: Dict[str, object],
    num_boost_round: int,
    early_stopping_rounds: int,
    objective_mode: str,
    train_group: list[int] | None = None,
    val_group: list[int] | None = None,
    train_rank_labels: np.ndarray | None = None,
    val_rank_labels: np.ndarray | None = None,
):
    if objective_mode == "regression":
        return train_booster(
            train_x,
            train_profit,
            val_x,
            val_profit,
            params=params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
    if objective_mode != "lambdarank":
        raise ValueError(f"unsupported objective mode: {objective_mode}")
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed; run: pip install -r requirements.txt") from exc

    if train_group is None or train_rank_labels is None:
        raise ValueError("lambdarank requires train_group and train_rank_labels")
    rank_params = params.copy()
    rank_params["objective"] = "lambdarank"
    rank_params["metric"] = "ndcg"
    rank_params.setdefault("eval_at", [1, 3, 10])

    train_set = lgb.Dataset(
        train_x,
        label=train_rank_labels,
        group=train_group,
        feature_name=list(train_x.columns),
    )
    valid_sets = None
    callbacks = [lgb.log_evaluation(100)]
    if val_x is not None and val_rank_labels is not None and val_group is not None:
        val_set = lgb.Dataset(
            val_x,
            label=val_rank_labels,
            group=val_group,
            feature_name=list(train_x.columns),
            reference=train_set,
        )
        valid_sets = [val_set]
        callbacks.insert(0, lgb.early_stopping(early_stopping_rounds))

    return lgb.train(
        rank_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def _add_prediction_columns(
    candidate_df: pd.DataFrame,
    pred_arrays: Sequence[np.ndarray],
    seeds: Sequence[int],
    score_col: str = "pred_window_profit",
) -> pd.DataFrame:
    out = candidate_df.copy()
    if len(pred_arrays) != len(seeds):
        raise ValueError("prediction arrays and seeds must have the same length")
    pred_matrix = np.vstack(pred_arrays)
    for seed, pred in zip(seeds, pred_arrays):
        out[f"{score_col}_seed{seed}"] = pred
    out[score_col] = np.mean(pred_matrix, axis=0)
    out[f"{score_col}_std"] = np.std(pred_matrix, axis=0)
    return out


def _add_rank_columns(
    candidate_df: pd.DataFrame,
    score_col: str = "pred_window_profit",
) -> pd.DataFrame:
    out = candidate_df.copy()
    out["pred_rank"] = out.groupby("date", sort=True)[score_col].rank(
        method="first",
        ascending=False,
    )
    if "true_window_profit" in out.columns:
        out["true_rank"] = out.groupby("date", sort=True)["true_window_profit"].rank(
            method="first",
            ascending=False,
        )

    margins = []
    for date, group in out.groupby("date", sort=True):
        ordered = group.sort_values(score_col, ascending=False)
        if len(ordered) >= 2:
            margin = float(ordered.iloc[0][score_col] - ordered.iloc[1][score_col])
        else:
            margin = 0.0
        margins.append({"date": date, "top1_minus_top2_margin": margin})
    margin_df = pd.DataFrame(margins)
    return out.merge(margin_df, on="date", how="left")


def _validation_day_metrics(
    candidate_df: pd.DataFrame,
    score_col: str = "pred_window_profit",
    label_col: str = "true_window_profit",
) -> pd.DataFrame:
    rows = []
    for date, group in candidate_df.groupby("date", sort=True):
        pred_ordered = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        true_ordered = group.sort_values(label_col, ascending=False).reset_index(drop=True)
        selected = pred_ordered.iloc[0]
        oracle = true_ordered.iloc[0]
        selected_profit = float(selected["true_window_profit"])
        selected_label_value = float(selected[label_col])
        oracle_label_value = float(oracle[label_col])
        oracle_window_profit = float(oracle["true_window_profit"])
        oracle_profit = max(0.0, oracle_window_profit)
        baseline_profit = float(selected.get("baseline_true_window_profit", np.nan))
        selected_delta_profit = float(selected.get("true_delta_profit", np.nan))
        oracle_delta_profit = float(oracle.get("true_delta_profit", np.nan))
        top3 = pred_ordered.head(3)
        top3_hit = bool(
            (
                (top3["charge_start"].astype(int) == int(oracle["charge_start"]))
                & (top3["discharge_start"].astype(int) == int(oracle["discharge_start"]))
            ).any()
        )
        top1_hit = bool(
            int(selected["charge_start"]) == int(oracle["charge_start"])
            and int(selected["discharge_start"]) == int(oracle["discharge_start"])
        )
        rows.append(
            {
                "date": date,
                "selected_profit": selected_profit,
                "selected_label_value": selected_label_value,
                "baseline_profit": baseline_profit,
                "selected_delta_profit": selected_delta_profit,
                "oracle_window_profit": oracle_window_profit,
                "oracle_label_value": oracle_label_value,
                "oracle_delta_profit": oracle_delta_profit,
                "oracle_profit": oracle_profit,
                "regret": oracle_profit - selected_profit,
                "delta_regret": oracle_label_value - selected_label_value,
                "capture_ratio": selected_profit / oracle_profit if oracle_profit > 0 else np.nan,
                "top1_window_hit": top1_hit,
                "top3_window_hit": top3_hit,
                "selected_charge_start": int(selected["charge_start"]),
                "selected_discharge_start": int(selected["discharge_start"]),
                "baseline_charge_start": int(selected.get("baseline_charge_start", -1)),
                "baseline_discharge_start": int(selected.get("baseline_discharge_start", -1)),
                "oracle_charge_start": int(oracle["charge_start"]),
                "oracle_discharge_start": int(oracle["discharge_start"]),
                "selected_pred_window_score": float(selected[score_col]),
                "selected_score_std": float(selected.get(f"{score_col}_std", 0.0)),
                "top1_minus_top2_margin": float(selected.get("top1_minus_top2_margin", 0.0)),
                "true_rank_of_selected": float(selected.get("true_rank", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def windows_to_submission(
    base_df: pd.DataFrame,
    selected: pd.DataFrame,
    price_values: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = base_df[[TIME_COL]].copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["__date__"] = df[TIME_COL].dt.date.astype(str)
    if price_values is not None:
        df["__price__"] = price_values
    selected_by_date = {row["date"]: row for row in selected.to_dict("records")}
    outputs = []
    meta = []
    for date, group in df.groupby("__date__", sort=True):
        group = group.sort_values(TIME_COL).reset_index(drop=True)
        if len(group) != 96:
            continue
        row = selected_by_date[date]
        power = np.zeros(96, dtype=float)
        tc = int(row["charge_start"])
        td = int(row["discharge_start"])
        power[tc : tc + 8] = -1000.0
        power[td : td + 8] = 1000.0
        price = (
            np.zeros(96, dtype=float)
            if price_values is None
            else group["__price__"].to_numpy(dtype=float)
        )
        outputs.append(
            pd.DataFrame(
                {
                    "times": group[TIME_COL].to_numpy(),
                    "实时价格": price,
                    "power": power,
                }
            )
        )
        meta.append(
            {
                "date": date,
                "pred_window_score": float(row["pred_window_profit"]),
                "score_std": float(row.get("pred_window_profit_std", 0.0)),
                "top1_top2_margin": float(row.get("top1_minus_top2_margin", 0.0)),
                "pred_rank": float(row.get("pred_rank", 1.0)),
                "true_rank": float(row.get("true_rank", np.nan)),
                "charge_start": tc,
                "discharge_start": td,
                "traded": True,
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct window-profit model for storage strategy.")
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--test-feature", default="")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--nwp-dir", default="")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features_train.csv")
    parser.add_argument("--test-nwp-cache", default="outputs/nwp_features_all.csv")
    parser.add_argument("--model-output", default="outputs/window_ranker_model.txt")
    parser.add_argument("--metadata-output", default="outputs/window_ranker_metadata.json")
    parser.add_argument("--val-window-output", default="outputs/val_windows_window_ranker.csv")
    parser.add_argument("--val-ranked-window-output", default="outputs/val_ranked_windows_window_ranker.csv")
    parser.add_argument("--val-day-metrics-output", default="outputs/val_window_ranker_day_metrics.csv")
    parser.add_argument("--test-window-output", default="outputs/test_windows_window_ranker.csv")
    parser.add_argument("--submission-output", default="outputs/output_window_ranker.csv")
    parser.add_argument("--meta-output", default="outputs/window_ranker_strategy_meta.csv")
    parser.add_argument("--val-days", type=int, default=59)
    parser.add_argument("--val-start-date", default="")
    parser.add_argument("--val-end-date", default="")
    parser.add_argument("--seeds", default="42,2024,2026")
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--objective-mode", choices=["regression", "lambdarank"], default="regression")
    parser.add_argument(
        "--label-mode",
        choices=["absolute", "baseline-delta"],
        default="absolute",
        help="Train on absolute window profit or profit delta versus a baseline daily window.",
    )
    parser.add_argument(
        "--baseline-window-score-col",
        default=DEFAULT_BASELINE_WINDOW_SCORE_COL,
        help="Window feature used to derive train/validation baseline windows when no baseline meta is provided.",
    )
    parser.add_argument("--train-baseline-meta", default="", help="Optional CSV with date,charge_start,discharge_start.")
    parser.add_argument("--val-baseline-meta", default="", help="Optional CSV with date,charge_start,discharge_start.")
    parser.add_argument(
        "--test-baseline-submission",
        default="",
        help="Optional submission CSV used as the test-period baseline for baseline-delta mode.",
    )
    parser.add_argument("--params-json", default="")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    if args.objective_mode == "regression":
        params["objective"] = "regression"
        params["metric"] = "rmse"
    if args.params_json:
        params.update(json.loads(args.params_json))
    seeds = parse_seeds(args.seeds)

    df = load_training_frame(
        args.train_feature,
        args.train_label,
        target_col=args.target_col,
        nwp_dir=args.nwp_dir,
        nwp_cache=args.nwp_cache,
    )
    train_df, val_df = split_by_day(
        df,
        val_ratio=0.2,
        val_start_date=args.val_start_date,
        val_end_date=args.val_end_date,
        val_days=args.val_days,
    )

    price_stats = fit_price_history_features(train_df, target_col=args.target_col)
    train_model_df = add_price_history_features(train_df, price_stats)
    val_model_df = add_price_history_features(val_df, price_stats)
    hist_stats = fit_history_stats(train_model_df, target_col=args.target_col)
    train_features = build_features(train_model_df, history_stats=hist_stats)
    val_features = build_features(val_model_df, history_stats=hist_stats)
    point_features = [col for col in DEFAULT_POINT_FEATURES if col in train_features.frame.columns]
    if not point_features:
        raise ValueError("no point features available for window ranker")

    train_windows, _ = build_window_dataset(
        train_df,
        train_features.frame,
        point_features,
        args.target_col,
        args.charge_start_min,
        args.charge_start_max,
        args.discharge_start_min,
        args.discharge_start_max,
        include_target=True,
    )
    val_windows, _ = build_window_dataset(
        val_df,
        val_features.frame,
        point_features,
        args.target_col,
        args.charge_start_min,
        args.charge_start_max,
        args.discharge_start_min,
        args.discharge_start_max,
        include_target=True,
    )
    if args.label_mode == "baseline-delta":
        train_baseline = prepare_baseline_windows(
            train_windows,
            score_col=args.baseline_window_score_col,
            meta_path=args.train_baseline_meta,
        )
        val_baseline = prepare_baseline_windows(
            val_windows,
            score_col=args.baseline_window_score_col,
            meta_path=args.val_baseline_meta,
        )
        train_windows = attach_baseline_window_context(train_windows, train_baseline)
        val_windows = attach_baseline_window_context(val_windows, val_baseline)
    label_col = "true_delta_profit" if args.label_mode == "baseline-delta" else "true_window_profit"
    feature_columns = [
        col for col in train_windows.columns if col not in {"date", *LABEL_COLUMNS}
    ]
    train_x = train_windows[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    val_x = val_windows[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_y = train_windows[label_col].to_numpy(dtype=float)
    val_y = val_windows[label_col].to_numpy(dtype=float)
    train_group = _group_sizes_by_date(train_windows)
    val_group = _group_sizes_by_date(val_windows)
    train_rank_labels = _ranker_labels(train_windows, target_col=label_col)
    val_rank_labels = _ranker_labels(val_windows, target_col=label_col)

    val_preds = []
    best_iterations = []
    for seed in seeds:
        model = _train_window_booster(
            train_x,
            train_y,
            val_x,
            val_y,
            params=params_for_seed(params, seed),
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            objective_mode=args.objective_mode,
            train_group=train_group,
            val_group=val_group,
            train_rank_labels=train_rank_labels,
            val_rank_labels=val_rank_labels,
        )
        best_iteration = int(model.best_iteration or args.num_boost_round)
        best_iterations.append(best_iteration)
        val_preds.append(model.predict(val_x, num_iteration=best_iteration))
    val_windows = _add_prediction_columns(val_windows, val_preds, seeds)
    val_windows = _add_rank_columns(val_windows)
    selected_val = choose_daily_windows(val_windows, "pred_window_profit")
    day_metrics_df = _validation_day_metrics(val_windows, label_col=label_col)
    val_submission, _ = windows_to_submission(
        val_df,
        selected_val,
        price_values=val_df[args.target_col].to_numpy(dtype=float),
    )
    joined = val_submission.merge(val_df[[TIME_COL, args.target_col]], on=TIME_COL, how="left")
    day_profit = []
    joined["date"] = pd.to_datetime(joined["times"]).dt.date
    for date, group in joined.groupby("date", sort=True):
        oracle = optimize_one_day(
            group[args.target_col].to_numpy(dtype=float),
            charge_start_min=args.charge_start_min,
            charge_start_max=args.charge_start_max,
            discharge_start_min=args.discharge_start_min,
            discharge_start_max=args.discharge_start_max,
        )
        profit = evaluate_power(group[args.target_col], group["power"])
        day_profit.append(
            {
                "date": str(date),
                "profit": profit,
                "oracle_profit": max(0.0, float(oracle.best_profit)),
            }
        )
    day_profit_df = pd.DataFrame(day_profit)
    day_metrics_df = day_metrics_df.merge(
        day_profit_df[["date", "profit"]].rename(columns={"profit": "submission_profit"}),
        on="date",
        how="left",
    )
    Path(args.val_window_output).parent.mkdir(parents=True, exist_ok=True)
    selected_val.to_csv(args.val_window_output, index=False)
    Path(args.val_ranked_window_output).parent.mkdir(parents=True, exist_ok=True)
    val_windows.to_csv(args.val_ranked_window_output, index=False)
    Path(args.val_day_metrics_output).parent.mkdir(parents=True, exist_ok=True)
    day_metrics_df.to_csv(args.val_day_metrics_output, index=False)
    avg_profit = float(day_metrics_df["selected_profit"].mean())
    avg_delta_profit = (
        float(day_metrics_df["selected_delta_profit"].mean())
        if "selected_delta_profit" in day_metrics_df.columns
        else float("nan")
    )
    positive_delta_rate = (
        float((day_metrics_df["selected_delta_profit"] > 0).mean())
        if "selected_delta_profit" in day_metrics_df.columns
        else float("nan")
    )
    total_oracle = float(day_metrics_df["oracle_profit"].sum())
    capture_ratio = float(day_metrics_df["selected_profit"].sum() / total_oracle) if total_oracle else 0.0
    avg_regret = float(day_metrics_df["regret"].mean())
    top1_hit = float(day_metrics_df["top1_window_hit"].mean())
    top3_hit = float(day_metrics_df["top3_window_hit"].mean())
    avg_margin = float(day_metrics_df["top1_minus_top2_margin"].mean())
    print(
        "window_ranker_validation="
        f"avg_profit={avg_profit:.6f}, "
        f"avg_delta_profit={avg_delta_profit:.6f}, "
        f"positive_delta_rate={positive_delta_rate:.6f}, "
        f"capture_ratio={capture_ratio:.6f}, "
        f"regret={avg_regret:.6f}, "
        f"top1_window_hit={top1_hit:.6f}, "
        f"top3_window_hit={top3_hit:.6f}, "
        f"top1_minus_top2_margin={avg_margin:.6f}"
    )

    full_price_stats = fit_price_history_features(df, target_col=args.target_col)
    full_model_df = add_price_history_features(df, full_price_stats)
    full_hist_stats = fit_history_stats(full_model_df, target_col=args.target_col)
    full_features = build_features(full_model_df, history_stats=full_hist_stats)
    full_point_features = [col for col in point_features if col in full_features.frame.columns]
    full_windows, _ = build_window_dataset(
        df,
        full_features.frame,
        full_point_features,
        args.target_col,
        args.charge_start_min,
        args.charge_start_max,
        args.discharge_start_min,
        args.discharge_start_max,
        include_target=True,
    )
    if args.label_mode == "baseline-delta":
        full_baseline = prepare_baseline_windows(
            full_windows,
            score_col=args.baseline_window_score_col,
            meta_path=args.train_baseline_meta,
        )
        full_windows = attach_baseline_window_context(full_windows, full_baseline)
    full_feature_columns = [
        col for col in full_windows.columns if col not in {"date", *LABEL_COLUMNS}
    ]
    full_x = full_windows[full_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    full_y = full_windows[label_col].to_numpy(dtype=float)
    full_group = _group_sizes_by_date(full_windows)
    full_rank_labels = _ranker_labels(full_windows, target_col=label_col)

    final_models = []
    model_paths = []
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    for seed, rounds in zip(seeds, best_iterations):
        model = _train_window_booster(
            full_x,
            full_y,
            None,
            None,
            params=params_for_seed(params, seed),
            num_boost_round=rounds,
            early_stopping_rounds=args.early_stopping_rounds,
            objective_mode=args.objective_mode,
            train_group=full_group,
            train_rank_labels=full_rank_labels,
        )
        path = Path(args.model_output)
        model_path = str(path.with_name(f"{path.stem}_seed{seed}{path.suffix}"))
        model.save_model(model_path)
        final_models.append(model)
        model_paths.append(model_path)

    Path(args.metadata_output).write_text(
        json.dumps(
            {
                "model_paths": model_paths,
                "seeds": seeds,
                "feature_columns": full_feature_columns,
                "point_features": full_point_features,
                "price_history_stats": full_price_stats,
                "history_stats": full_hist_stats,
                "best_iterations": best_iterations,
                "objective_mode": args.objective_mode,
                "label_mode": args.label_mode,
                "label_col": label_col,
                "baseline_window_score_col": args.baseline_window_score_col,
                "validation_metrics": {
                    "avg_profit": avg_profit,
                    "avg_delta_profit": avg_delta_profit,
                    "positive_delta_rate": positive_delta_rate,
                    "capture_ratio": capture_ratio,
                    "regret": avg_regret,
                    "top1_window_hit": top1_hit,
                    "top3_window_hit": top3_hit,
                    "top1_minus_top2_margin": avg_margin,
                    "day_metrics_output": args.val_day_metrics_output,
                    "selected_window_output": args.val_window_output,
                    "ranked_window_output": args.val_ranked_window_output,
                },
                "strategy_constraints": {
                    "charge_start_min": args.charge_start_min,
                    "charge_start_max": args.charge_start_max,
                    "discharge_start_min": args.discharge_start_min,
                    "discharge_start_max": args.discharge_start_max,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.test_feature:
        test_df = pd.read_csv(args.test_feature)
        test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
        if args.nwp_dir:
            nwp = load_or_build_nwp_features(
                args.nwp_dir,
                args.test_nwp_cache,
                start_time=str(test_df[TIME_COL].min()),
                end_time=str(test_df[TIME_COL].max()),
            )
            test_df = merge_nwp_features(test_df, nwp)
        test_model_df = add_price_history_features(test_df, full_price_stats)
        test_features = build_features(test_model_df, history_stats=full_hist_stats)
        test_windows, _ = build_window_dataset(
            test_df,
            test_features.frame,
            full_point_features,
            args.target_col,
            args.charge_start_min,
            args.charge_start_max,
            args.discharge_start_min,
            args.discharge_start_max,
            include_target=False,
        )
        if args.label_mode == "baseline-delta":
            if not args.test_baseline_submission:
                raise ValueError("--test-baseline-submission is required when --label-mode baseline-delta and --test-feature is used")
            test_baseline = prepare_baseline_windows(
                test_windows,
                score_col=args.baseline_window_score_col,
                submission_path=args.test_baseline_submission,
            )
            test_windows = attach_baseline_window_context(test_windows, test_baseline)
        test_x = test_windows[full_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_preds = [
            model.predict(test_x, num_iteration=rounds)
            for model, rounds in zip(final_models, best_iterations)
        ]
        test_windows = _add_prediction_columns(test_windows, test_preds, seeds)
        test_windows = _add_rank_columns(test_windows)
        selected_test = choose_daily_windows(test_windows, "pred_window_profit")
        Path(args.test_window_output).parent.mkdir(parents=True, exist_ok=True)
        test_windows.to_csv(args.test_window_output, index=False)
        submission, meta = windows_to_submission(test_df, selected_test)
        submission.to_csv(args.submission_output, index=False)
        meta.to_csv(args.meta_output, index=False)
        print(f"saved_window_ranker_submission={args.submission_output}, rows={len(submission)}")


if __name__ == "__main__":
    main()
