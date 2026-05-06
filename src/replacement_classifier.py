from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from .train_window_ranker import LABEL_COLUMNS, _add_rank_columns


CLASSIFIER_PARAMS: Dict[str, object] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_data_in_leaf": 48,
    "seed": 42,
    "verbose": -1,
}

PRED_PROBA_COL = "pred_positive_proba"
PRED_EXPECTED_DELTA_COL = "pred_expected_delta"
POSITIVE_LABEL_COL = "positive_delta_label"
TOPK_SCORE_COL = "daily_topk_prior_score"
RISK_PROBA_COL = "risk_positive_proba"
RISK_EXPECTED_DELTA_COL = "risk_expected_delta"
RULE_RISK_SCORE_COL = "rule_risk_score"
BASELINE_STABILITY_PASS_COL = "baseline_stability_pass"
BASELINE_STABILITY_MAX_ABS_DELTA_COL = "baseline_stability_max_abs_delta"
DISCHARGE_SPIKE_RISK_COL = "discharge_move_spike_risk_proxy"
DISCHARGE_PLATEAU_STRENGTH_COL = "discharge_move_plateau_strength_proxy"

DEFAULT_TOPK_PRIOR_COLS = [
    "delta_vs_baseline_spread_price_hist_recent_28d_slot_mean",
    "delta_vs_baseline_spread_price_hist_month_slot_median",
    "delta_vs_baseline_spread_hist_slot_mean",
    "delta_vs_baseline_spread_hist_month_slot_mean",
    "delta_vs_baseline_spread_price_hist_same_month_day_slot",
]


@dataclass
class DeltaCalibrator:
    bins: list[dict[str, float]]
    default_expected_delta: float

    def predict_expected_delta(self, probabilities: Sequence[float]) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=float)
        if not self.bins:
            return np.full(len(probs), self.default_expected_delta, dtype=float)
        centers = np.asarray([row["mean_proba"] for row in self.bins], dtype=float)
        means = np.asarray([row["mean_delta"] for row in self.bins], dtype=float)
        out = np.empty(len(probs), dtype=float)
        for i, proba in enumerate(probs):
            idx = int(np.argmin(np.abs(centers - proba)))
            out[i] = means[idx]
        return out

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "default_expected_delta": self.default_expected_delta,
            "bins": self.bins,
        }


def filter_near_baseline_windows(
    windows: pd.DataFrame,
    max_shift: int,
    include_same_window: bool = False,
) -> pd.DataFrame:
    required = {
        "abs_delta_charge_start",
        "abs_delta_discharge_start",
        "same_as_baseline_window",
    }
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"windows missing baseline-delta columns: {sorted(missing)}")
    out = windows.loc[
        (windows["abs_delta_charge_start"].astype(float) <= float(max_shift))
        & (windows["abs_delta_discharge_start"].astype(float) <= float(max_shift))
    ].copy()
    if not include_same_window:
        out = out.loc[out["same_as_baseline_window"].astype(int) == 0].copy()
    if out.empty:
        raise ValueError(f"no near-baseline windows available for max_shift={max_shift}")
    return out.reset_index(drop=True)


def _rank_pct_by_date(
    frame: pd.DataFrame,
    col: str,
    ascending: bool = False,
) -> pd.Series:
    rank = frame.groupby("date", sort=True)[col].rank(
        method="average",
        ascending=ascending,
    )
    count = frame.groupby("date", sort=True)[col].transform("count").astype(float)
    denom = (count - 1.0).replace(0.0, np.nan)
    score = 1.0 - (rank.astype(float) - 1.0) / denom
    return score.fillna(1.0).clip(lower=0.0, upper=1.0)


def _safe_max_scale(values: pd.Series) -> float:
    finite = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 1.0
    value = float(finite.max())
    return value if value > 0.0 else 1.0


def add_replacement_calibration_features(
    windows: pd.DataFrame,
    topk_prior_cols: Sequence[str] | None = None,
    topk_score_col: str = TOPK_SCORE_COL,
) -> pd.DataFrame:
    required = {
        "date",
        "delta_charge_start",
        "delta_discharge_start",
        "abs_delta_charge_start",
        "abs_delta_discharge_start",
        "gap_slots",
        "baseline_gap_slots",
        "delta_gap_slots",
    }
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"windows missing calibration columns: {sorted(missing)}")

    out = windows.copy()
    out["date"] = out["date"].astype(str)
    out["total_abs_shift"] = (
        out["abs_delta_charge_start"].astype(float)
        + out["abs_delta_discharge_start"].astype(float)
    )
    out["max_abs_shift"] = np.maximum(
        out["abs_delta_charge_start"].astype(float),
        out["abs_delta_discharge_start"].astype(float),
    )
    out["shift_balance_abs_diff"] = (
        out["abs_delta_charge_start"].astype(float)
        - out["abs_delta_discharge_start"].astype(float)
    ).abs()
    out["same_direction_shift"] = (
        np.sign(out["delta_charge_start"].astype(float))
        == np.sign(out["delta_discharge_start"].astype(float))
    ).astype(int)
    out["opposite_direction_shift"] = (
        np.sign(out["delta_charge_start"].astype(float))
        == -np.sign(out["delta_discharge_start"].astype(float))
    ).astype(int)
    out["charge_shift_sign"] = np.sign(out["delta_charge_start"].astype(float))
    out["discharge_shift_sign"] = np.sign(out["delta_discharge_start"].astype(float))
    out["candidate_gap_delta_abs"] = out["delta_gap_slots"].astype(float).abs()
    out["baseline_gap_strength"] = out["baseline_gap_slots"].astype(float)
    out["shift_penalty"] = out["total_abs_shift"] / _safe_max_scale(out["total_abs_shift"])
    out["shift_penalty"] = out["shift_penalty"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = add_discharge_shape_risk_features(out)

    prior_cols = [col for col in (topk_prior_cols or DEFAULT_TOPK_PRIOR_COLS) if col in out.columns]
    if not prior_cols:
        delta_cols = [
            col
            for col in out.columns
            if col.startswith("delta_vs_baseline_spread_")
            and "true" not in col.lower()
        ]
        prior_cols = delta_cols[:5]
    if not prior_cols:
        out[topk_score_col] = -out["total_abs_shift"].astype(float)
        out[f"{topk_score_col}_rank"] = _rank_pct_by_date(out, topk_score_col, ascending=False)
        out[f"{topk_score_col}_pct_rank"] = out[f"{topk_score_col}_rank"]
        return out

    rank_cols: list[str] = []
    z_cols: list[str] = []
    for col in prior_cols:
        values = out[col].astype(float)
        rank_col = f"{col}_daily_pct_rank"
        centered_col = f"{col}_daily_centered"
        z_col = f"{col}_daily_z"
        out[rank_col] = _rank_pct_by_date(out, col, ascending=False)
        daily_mean = values.groupby(out["date"], sort=True).transform("mean")
        daily_std = values.groupby(out["date"], sort=True).transform("std").replace(0, np.nan)
        out[centered_col] = values - daily_mean
        out[z_col] = (values - daily_mean) / daily_std
        out[z_col] = out[z_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rank_cols.append(rank_col)
        z_cols.append(z_col)

    out["prior_rank_mean"] = out[rank_cols].mean(axis=1)
    out["prior_rank_min"] = out[rank_cols].min(axis=1)
    out["prior_rank_max"] = out[rank_cols].max(axis=1)
    out["prior_z_mean"] = out[z_cols].mean(axis=1)
    out["prior_z_min"] = out[z_cols].min(axis=1)
    out["prior_z_max"] = out[z_cols].max(axis=1)
    out[topk_score_col] = (
        out["prior_rank_mean"].astype(float)
        + 0.05 * out["prior_rank_min"].astype(float)
        + 0.02 * out["prior_z_mean"].clip(lower=-5.0, upper=5.0).astype(float)
        - 0.10 * out["shift_penalty"].astype(float)
        - 0.03
        * out["candidate_gap_delta_abs"].astype(float)
        / _safe_max_scale(out["candidate_gap_delta_abs"])
    )
    out[topk_score_col] = out[topk_score_col].replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    out[f"{topk_score_col}_rank"] = out.groupby("date", sort=True)[topk_score_col].rank(
        method="first",
        ascending=False,
    )
    out[f"{topk_score_col}_pct_rank"] = _rank_pct_by_date(out, topk_score_col, ascending=False)

    margin_rows: list[dict[str, float | str]] = []
    for date, group in out.groupby("date", sort=True):
        ordered = group.sort_values(topk_score_col, ascending=False)
        best = float(ordered.iloc[0][topk_score_col])
        second = float(ordered.iloc[1][topk_score_col]) if len(ordered) >= 2 else best
        median = float(ordered[topk_score_col].median())
        margin_rows.append(
            {
                "date": str(date),
                "daily_topk_prior_top1": best,
                "daily_topk_prior_top2": second,
                "daily_topk_prior_top1_minus_top2": best - second,
                "daily_topk_prior_top1_minus_median": best - median,
            }
        )
    out = out.merge(pd.DataFrame(margin_rows), on="date", how="left")
    out["daily_topk_prior_score_minus_daily_top1"] = (
        out[topk_score_col].astype(float) - out["daily_topk_prior_top1"].astype(float)
    )
    return out


def add_discharge_shape_risk_features(windows: pd.DataFrame) -> pd.DataFrame:
    out = windows.copy()

    move = out["delta_discharge_start"].astype(float) if "delta_discharge_start" in out.columns else 0.0
    moved_later = move > 0.0
    moved_earlier = move < 0.0
    moved = move != 0.0

    feature_pairs = [
        ("price_hist_recent_28d_slot_mean", "recent28_price"),
        ("hist_slot_mean", "hist_slot"),
        ("hist_month_slot_mean", "hist_month_slot"),
        ("net_load", "net_load"),
        ("net_load_intertie_plus", "net_load_intertie_plus"),
        ("nwp_tcc_mean", "nwp_tcc"),
        ("nwp_wind_speed_mean", "nwp_wind"),
    ]

    risk_terms: list[pd.Series] = []
    plateau_terms: list[pd.Series] = []
    for base_col, tag in feature_pairs:
        baseline_mean_col = f"baseline_discharge_{base_col}"
        candidate_mean_col = f"discharge_{base_col}"
        baseline_max_col = f"baseline_discharge_{base_col}_max"
        baseline_min_col = f"baseline_discharge_{base_col}_min"
        candidate_max_col = f"discharge_{base_col}_max"
        candidate_min_col = f"discharge_{base_col}_min"
        baseline_std_col = f"baseline_discharge_{base_col}_std"
        candidate_std_col = f"discharge_{base_col}_std"

        required = {
            baseline_mean_col,
            candidate_mean_col,
            baseline_max_col,
            baseline_min_col,
            candidate_max_col,
            candidate_min_col,
            baseline_std_col,
            candidate_std_col,
        }
        if not required.issubset(out.columns):
            continue

        baseline_mean = out[baseline_mean_col].astype(float)
        candidate_mean = out[candidate_mean_col].astype(float)
        baseline_max = out[baseline_max_col].astype(float)
        baseline_min = out[baseline_min_col].astype(float)
        candidate_max = out[candidate_max_col].astype(float)
        candidate_min = out[candidate_min_col].astype(float)
        baseline_std = out[baseline_std_col].astype(float)
        candidate_std = out[candidate_std_col].astype(float)

        spike_risk = np.where(
            moved_later,
            baseline_max - candidate_mean,
            np.where(moved_earlier, candidate_max - baseline_mean, 0.0),
        )
        plateau_strength = np.where(
            moved_later,
            candidate_min - baseline_mean,
            np.where(moved_earlier, baseline_min - candidate_mean, 0.0),
        )
        mean_delta = candidate_mean - baseline_mean
        std_delta = candidate_std - baseline_std

        risk_col = f"discharge_move_{tag}_spike_risk"
        plateau_col = f"discharge_move_{tag}_plateau_strength"
        mean_delta_col = f"discharge_move_{tag}_mean_delta"
        std_delta_col = f"discharge_move_{tag}_std_delta"
        max_delta_col = f"discharge_move_{tag}_max_delta"
        min_delta_col = f"discharge_move_{tag}_min_delta"

        out[risk_col] = pd.Series(spike_risk, index=out.index).where(moved, 0.0)
        out[plateau_col] = pd.Series(plateau_strength, index=out.index).where(moved, 0.0)
        out[mean_delta_col] = mean_delta.where(moved, 0.0)
        out[std_delta_col] = std_delta.where(moved, 0.0)
        out[max_delta_col] = (candidate_max - baseline_max).where(moved, 0.0)
        out[min_delta_col] = (candidate_min - baseline_min).where(moved, 0.0)

        risk_terms.append(out[risk_col].astype(float))
        plateau_terms.append(out[plateau_col].astype(float))

    if risk_terms:
        out[DISCHARGE_SPIKE_RISK_COL] = pd.concat(risk_terms, axis=1).max(axis=1)
        out[DISCHARGE_PLATEAU_STRENGTH_COL] = pd.concat(plateau_terms, axis=1).max(axis=1)
    else:
        out[DISCHARGE_SPIKE_RISK_COL] = 0.0
        out[DISCHARGE_PLATEAU_STRENGTH_COL] = 0.0
    out["discharge_shape_risk_balance"] = (
        out[DISCHARGE_PLATEAU_STRENGTH_COL].astype(float)
        - out[DISCHARGE_SPIKE_RISK_COL].astype(float)
    )
    return out


def filter_daily_topk_replacement_candidates(
    windows: pd.DataFrame,
    daily_top_k: int,
    score_col: str = TOPK_SCORE_COL,
) -> pd.DataFrame:
    if int(daily_top_k) <= 0:
        return windows.reset_index(drop=True)
    if score_col not in windows.columns:
        raise ValueError(f"top-k score column not found: {score_col}")
    out = (
        windows.sort_values(["date", score_col], ascending=[True, False])
        .groupby("date", sort=True, as_index=False)
        .head(int(daily_top_k))
        .copy()
    )
    if out.empty:
        raise ValueError(f"daily top-k filtering removed all rows: k={daily_top_k}")
    return out.reset_index(drop=True)


def baseline_meta_from_attached_windows(windows: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "baseline_charge_start", "baseline_discharge_start"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"windows missing attached baseline columns: {sorted(missing)}")
    out = (
        windows.loc[:, ["date", "baseline_charge_start", "baseline_discharge_start"]]
        .copy()
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["date"] = out["date"].astype(str)
    out["baseline_charge_start"] = out["baseline_charge_start"].astype(int)
    out["baseline_discharge_start"] = out["baseline_discharge_start"].astype(int)
    return out


def normalize_baseline_meta(reference: pd.DataFrame) -> pd.DataFrame:
    out = reference.copy()
    if {"baseline_charge_start", "baseline_discharge_start"}.issubset(out.columns):
        pass
    elif {"charge_start", "discharge_start"}.issubset(out.columns):
        out = out.rename(
            columns={
                "charge_start": "baseline_charge_start",
                "discharge_start": "baseline_discharge_start",
            }
        )
    else:
        raise ValueError(
            "baseline reference must contain baseline_charge_start/baseline_discharge_start "
            "or charge_start/discharge_start"
        )
    return baseline_meta_from_attached_windows(out)


def add_baseline_stability_features(
    windows: pd.DataFrame,
    reference_baseline: pd.DataFrame,
    max_abs_delta: int,
) -> pd.DataFrame:
    required = {"date", "baseline_charge_start", "baseline_discharge_start"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"windows missing baseline columns for stability gate: {sorted(missing)}")

    reference = normalize_baseline_meta(reference_baseline).rename(
        columns={
            "baseline_charge_start": "baseline_stability_reference_charge_start",
            "baseline_discharge_start": "baseline_stability_reference_discharge_start",
        }
    )
    out = windows.copy()
    out["date"] = out["date"].astype(str)
    out = out.merge(reference, on="date", how="left")
    missing_reference = out["baseline_stability_reference_charge_start"].isna() | out[
        "baseline_stability_reference_discharge_start"
    ].isna()
    out["baseline_stability_charge_delta"] = (
        out["baseline_charge_start"].astype(float)
        - out["baseline_stability_reference_charge_start"].astype(float)
    )
    out["baseline_stability_discharge_delta"] = (
        out["baseline_discharge_start"].astype(float)
        - out["baseline_stability_reference_discharge_start"].astype(float)
    )
    out["baseline_stability_abs_charge_delta"] = out["baseline_stability_charge_delta"].abs()
    out["baseline_stability_abs_discharge_delta"] = out["baseline_stability_discharge_delta"].abs()
    out[BASELINE_STABILITY_MAX_ABS_DELTA_COL] = out[
        ["baseline_stability_abs_charge_delta", "baseline_stability_abs_discharge_delta"]
    ].max(axis=1)
    out["baseline_stability_total_abs_delta"] = (
        out["baseline_stability_abs_charge_delta"] + out["baseline_stability_abs_discharge_delta"]
    )
    out[BASELINE_STABILITY_PASS_COL] = (
        (~missing_reference)
        & (out[BASELINE_STABILITY_MAX_ABS_DELTA_COL].astype(float) <= float(max_abs_delta))
    ).astype(float)
    out.loc[missing_reference, BASELINE_STABILITY_MAX_ABS_DELTA_COL] = 999999.0
    out.loc[missing_reference, "baseline_stability_total_abs_delta"] = 999999.0
    return out


def prepare_replacement_candidates(
    windows: pd.DataFrame,
    max_shift: int,
    daily_top_k: int = 0,
    include_same_window: bool = False,
    topk_score_col: str = TOPK_SCORE_COL,
) -> pd.DataFrame:
    near = filter_near_baseline_windows(
        windows,
        max_shift=max_shift,
        include_same_window=include_same_window,
    )
    calibrated = add_replacement_calibration_features(
        near,
        topk_score_col=topk_score_col,
    )
    return filter_daily_topk_replacement_candidates(
        calibrated,
        daily_top_k=daily_top_k,
        score_col=topk_score_col,
    )


def drop_attached_baseline_columns(windows: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col
        for col in windows.columns
        if col.startswith("baseline_")
        or col.startswith("delta_vs_baseline_")
        or col in {
            "delta_charge_start",
            "delta_discharge_start",
            "abs_delta_charge_start",
            "abs_delta_discharge_start",
            "delta_gap_slots",
            "same_as_baseline_window",
            "true_delta_profit",
        }
    ]
    return windows.drop(columns=drop_cols, errors="ignore")


def predict_binary(model: Any, x: pd.DataFrame, num_iteration: int) -> np.ndarray:
    pred = np.asarray(model.predict(x, num_iteration=num_iteration))
    if pred.ndim == 2:
        if pred.shape[1] == 2:
            return pred[:, 1]
        return pred.reshape(-1)
    return pred.reshape(-1)


def replacement_feature_columns(windows: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        *LABEL_COLUMNS,
        POSITIVE_LABEL_COL,
        PRED_PROBA_COL,
        f"{PRED_PROBA_COL}_std",
        PRED_EXPECTED_DELTA_COL,
        "pred_window_profit",
        "pred_window_profit_std",
        "pred_rank",
        "true_rank",
        "top1_minus_top2_margin",
        "daily_topk_prior_top1",
        "daily_topk_prior_top2",
    }
    return [col for col in windows.columns if col not in excluded]


def risk_feature_columns(windows: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        *LABEL_COLUMNS,
        POSITIVE_LABEL_COL,
        RISK_PROBA_COL,
        f"{RISK_PROBA_COL}_std",
        RISK_EXPECTED_DELTA_COL,
        "true_rank",
    }
    return [col for col in windows.columns if col not in excluded]


def add_positive_delta_label(
    windows: pd.DataFrame,
    positive_delta_threshold: float = 0.0,
) -> pd.DataFrame:
    if "true_delta_profit" not in windows.columns:
        raise ValueError("true_delta_profit is required for classifier training")
    out = windows.copy()
    out[POSITIVE_LABEL_COL] = (
        out["true_delta_profit"].astype(float) > float(positive_delta_threshold)
    ).astype(int)
    return out


def _stage1_eligible_rows(
    group: pd.DataFrame,
    proba_threshold: float,
    min_expected_delta: float,
    min_margin: float,
    max_proba_std: float | None,
    require_baseline_stability: bool = False,
) -> pd.DataFrame:
    eligible = group.loc[
        (group[PRED_PROBA_COL].astype(float) >= float(proba_threshold))
        & (group[PRED_EXPECTED_DELTA_COL].astype(float) >= float(min_expected_delta))
        & (group["top1_minus_top2_margin"].astype(float) >= float(min_margin))
    ].copy()
    if require_baseline_stability:
        if BASELINE_STABILITY_PASS_COL not in eligible.columns:
            raise ValueError("baseline stability gate requested but scored windows are missing stability features")
        eligible = eligible.loc[eligible[BASELINE_STABILITY_PASS_COL].astype(float) >= 1.0].copy()
    if max_proba_std is not None:
        eligible = eligible.loc[
            eligible[f"{PRED_PROBA_COL}_std"].astype(float) <= float(max_proba_std)
        ].copy()
    return eligible


def select_stage1_candidate_rows(
    scored_windows: pd.DataFrame,
    proba_threshold: float,
    min_expected_delta: float,
    min_margin: float = 0.0,
    max_proba_std: float | None = None,
    require_baseline_stability: bool = False,
) -> pd.DataFrame:
    rows = []
    for _, group in scored_windows.groupby("date", sort=True):
        eligible = _stage1_eligible_rows(
            group,
            proba_threshold=proba_threshold,
            min_expected_delta=min_expected_delta,
            min_margin=min_margin,
            max_proba_std=max_proba_std,
            require_baseline_stability=require_baseline_stability,
        )
        if eligible.empty:
            continue
        selected = eligible.sort_values(
            [PRED_EXPECTED_DELTA_COL, PRED_PROBA_COL, "top1_minus_top2_margin"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(selected)
    if not rows:
        return scored_windows.head(0).copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def fit_delta_calibrator(
    probabilities: Sequence[float],
    true_delta: Sequence[float],
    n_bins: int = 10,
) -> DeltaCalibrator:
    probs = np.asarray(probabilities, dtype=float)
    deltas = np.asarray(true_delta, dtype=float)
    if len(probs) != len(deltas):
        raise ValueError("probabilities and true_delta must have equal length")
    if len(probs) == 0:
        return DeltaCalibrator(bins=[], default_expected_delta=0.0)

    order = np.argsort(probs)
    chunks = np.array_split(order, max(1, min(int(n_bins), len(order))))
    bins: list[dict[str, float]] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        chunk_probs = probs[chunk]
        chunk_deltas = deltas[chunk]
        bins.append(
            {
                "min_proba": float(np.min(chunk_probs)),
                "max_proba": float(np.max(chunk_probs)),
                "mean_proba": float(np.mean(chunk_probs)),
                "count": float(len(chunk)),
                "positive_rate": float(np.mean(chunk_deltas > 0.0)),
                "mean_delta": float(np.mean(chunk_deltas)),
                "median_delta": float(np.median(chunk_deltas)),
            }
        )
    return DeltaCalibrator(
        bins=bins,
        default_expected_delta=float(np.mean(deltas)),
    )


def add_classifier_predictions(
    windows: pd.DataFrame,
    pred_arrays: Sequence[np.ndarray],
    seeds: Sequence[int],
    calibrator: DeltaCalibrator,
) -> pd.DataFrame:
    if len(pred_arrays) != len(seeds):
        raise ValueError("prediction arrays and seeds must have equal length")
    out = windows.copy()
    pred_matrix = np.vstack([np.asarray(pred, dtype=float).reshape(-1) for pred in pred_arrays])
    for seed, pred in zip(seeds, pred_arrays):
        out[f"{PRED_PROBA_COL}_seed{seed}"] = np.asarray(pred, dtype=float).reshape(-1)
    out[PRED_PROBA_COL] = np.mean(pred_matrix, axis=0)
    out[f"{PRED_PROBA_COL}_std"] = np.std(pred_matrix, axis=0)
    out[PRED_EXPECTED_DELTA_COL] = calibrator.predict_expected_delta(out[PRED_PROBA_COL].to_numpy())
    out["pred_window_profit"] = out[PRED_EXPECTED_DELTA_COL]
    out["pred_window_profit_std"] = out[f"{PRED_PROBA_COL}_std"]
    return _add_rank_columns(out, score_col="pred_window_profit")


def add_risk_predictions_to_scored_windows(
    scored_windows: pd.DataFrame,
    stage1_rows: pd.DataFrame,
    pred_arrays: Sequence[np.ndarray],
    seeds: Sequence[int],
    calibrator: DeltaCalibrator,
) -> pd.DataFrame:
    if len(pred_arrays) != len(seeds):
        raise ValueError("risk prediction arrays and seeds must have equal length")
    out = scored_windows.copy()
    out[RISK_PROBA_COL] = 0.0
    out[f"{RISK_PROBA_COL}_std"] = 0.0
    out[RISK_EXPECTED_DELTA_COL] = 0.0
    if stage1_rows.empty:
        return out

    pred_matrix = np.vstack([np.asarray(pred, dtype=float).reshape(-1) for pred in pred_arrays])
    risk_rows = stage1_rows[["date", "charge_start", "discharge_start"]].copy()
    for seed, pred in zip(seeds, pred_arrays):
        risk_rows[f"{RISK_PROBA_COL}_seed{seed}"] = np.asarray(pred, dtype=float).reshape(-1)
    risk_rows[RISK_PROBA_COL] = np.mean(pred_matrix, axis=0)
    risk_rows[f"{RISK_PROBA_COL}_std"] = np.std(pred_matrix, axis=0)
    risk_rows[RISK_EXPECTED_DELTA_COL] = calibrator.predict_expected_delta(
        risk_rows[RISK_PROBA_COL].to_numpy()
    )

    out = out.merge(
        risk_rows,
        on=["date", "charge_start", "discharge_start"],
        how="left",
        suffixes=("", "_risk_new"),
    )
    for col in [RISK_PROBA_COL, f"{RISK_PROBA_COL}_std", RISK_EXPECTED_DELTA_COL]:
        new_col = f"{col}_risk_new"
        if new_col in out.columns:
            out[col] = out[new_col].fillna(out[col])
            out = out.drop(columns=[new_col])
    return out


def add_risk_regression_predictions_to_scored_windows(
    scored_windows: pd.DataFrame,
    stage1_rows: pd.DataFrame,
    pred_arrays: Sequence[np.ndarray],
    seeds: Sequence[int],
) -> pd.DataFrame:
    if len(pred_arrays) != len(seeds):
        raise ValueError("risk regression prediction arrays and seeds must have equal length")
    out = scored_windows.copy()
    out[RISK_PROBA_COL] = 0.0
    out[f"{RISK_PROBA_COL}_std"] = 0.0
    out[RISK_EXPECTED_DELTA_COL] = 0.0
    out[f"{RISK_EXPECTED_DELTA_COL}_std"] = 0.0
    if stage1_rows.empty:
        return out

    pred_matrix = np.vstack([np.asarray(pred, dtype=float).reshape(-1) for pred in pred_arrays])
    risk_rows = stage1_rows[["date", "charge_start", "discharge_start"]].copy()
    for seed, pred in zip(seeds, pred_arrays):
        risk_rows[f"{RISK_EXPECTED_DELTA_COL}_seed{seed}"] = np.asarray(pred, dtype=float).reshape(-1)
    risk_rows[RISK_EXPECTED_DELTA_COL] = np.mean(pred_matrix, axis=0)
    risk_rows[f"{RISK_EXPECTED_DELTA_COL}_std"] = np.std(pred_matrix, axis=0)
    # This is a monotonic confidence proxy, not a calibrated probability.
    risk_rows[RISK_PROBA_COL] = np.mean(pred_matrix > 0.0, axis=0)
    risk_rows[f"{RISK_PROBA_COL}_std"] = np.std(pred_matrix > 0.0, axis=0)

    out = out.merge(
        risk_rows,
        on=["date", "charge_start", "discharge_start"],
        how="left",
        suffixes=("", "_risk_new"),
    )
    for col in [
        RISK_PROBA_COL,
        f"{RISK_PROBA_COL}_std",
        RISK_EXPECTED_DELTA_COL,
        f"{RISK_EXPECTED_DELTA_COL}_std",
    ]:
        new_col = f"{col}_risk_new"
        if new_col in out.columns:
            out[col] = out[new_col].fillna(out[col])
            out = out.drop(columns=[new_col])
    return out


def fit_rule_risk_gate(
    stage1_rows: pd.DataFrame,
    net_load_quantile: float = 0.85,
    hist_centered_quantile: float = 0.80,
    spike_risk_quantile: float = 0.50,
    plateau_strength_quantile: float = 0.50,
) -> dict[str, float]:
    required = {
        "delta_vs_baseline_spread_net_load",
        "delta_vs_baseline_spread_hist_slot_mean_daily_centered",
        DISCHARGE_SPIKE_RISK_COL,
        DISCHARGE_PLATEAU_STRENGTH_COL,
    }
    missing = required.difference(stage1_rows.columns)
    if missing:
        raise ValueError(f"stage1 rows missing rule-risk columns: {sorted(missing)}")
    if stage1_rows.empty:
        return {
            "net_load_threshold": float("inf"),
            "hist_centered_threshold": float("inf"),
            "net_load_quantile": float(net_load_quantile),
            "hist_centered_quantile": float(hist_centered_quantile),
            "spike_risk_threshold": float("-inf"),
            "plateau_strength_threshold": float("inf"),
            "spike_risk_quantile": float(spike_risk_quantile),
            "plateau_strength_quantile": float(plateau_strength_quantile),
            "min_shape_balance": 0.0,
            "require_both_windows_moved": True,
            "block_both_windows_earlier": True,
            "use_discharge_shape_gate": True,
        }
    positive_stage1 = stage1_rows.loc[
        stage1_rows.get(POSITIVE_LABEL_COL, 0).astype(int) > 0
    ].copy()
    if positive_stage1.empty:
        positive_stage1 = stage1_rows
    return {
        "net_load_threshold": float(
            stage1_rows["delta_vs_baseline_spread_net_load"].astype(float).quantile(net_load_quantile)
        ),
        "hist_centered_threshold": float(
            stage1_rows["delta_vs_baseline_spread_hist_slot_mean_daily_centered"]
            .astype(float)
            .quantile(hist_centered_quantile)
        ),
        "spike_risk_threshold": float(
            positive_stage1[DISCHARGE_SPIKE_RISK_COL].astype(float).quantile(spike_risk_quantile)
        ),
        "plateau_strength_threshold": float(
            positive_stage1[DISCHARGE_PLATEAU_STRENGTH_COL]
            .astype(float)
            .quantile(plateau_strength_quantile)
        ),
        "net_load_quantile": float(net_load_quantile),
        "hist_centered_quantile": float(hist_centered_quantile),
        "spike_risk_quantile": float(spike_risk_quantile),
        "plateau_strength_quantile": float(plateau_strength_quantile),
        "min_shape_balance": 0.0,
        "require_both_windows_moved": True,
        "block_both_windows_earlier": True,
        "use_discharge_shape_gate": True,
    }


def add_rule_risk_predictions_to_scored_windows(
    scored_windows: pd.DataFrame,
    gate: dict[str, float],
) -> pd.DataFrame:
    out = scored_windows.copy()
    net = out["delta_vs_baseline_spread_net_load"].astype(float)
    hist = out["delta_vs_baseline_spread_hist_slot_mean_daily_centered"].astype(float)
    net_thr = float(gate["net_load_threshold"])
    hist_thr = float(gate["hist_centered_threshold"])
    spike = out[DISCHARGE_SPIKE_RISK_COL].astype(float)
    plateau = out[DISCHARGE_PLATEAU_STRENGTH_COL].astype(float)
    spike_thr = float(gate.get("spike_risk_threshold", float("inf")))
    plateau_thr = float(gate.get("plateau_strength_threshold", float("-inf")))
    shape_balance = out["discharge_shape_risk_balance"].astype(float)
    shape_balance_thr = float(gate.get("min_shape_balance", 0.0))
    out["risk_rule_net_load_margin"] = net - net_thr
    out["risk_rule_hist_centered_margin"] = hist - hist_thr
    out["risk_rule_discharge_spike_margin"] = spike_thr - spike
    out["risk_rule_discharge_plateau_margin"] = plateau - plateau_thr
    out["risk_rule_discharge_shape_balance_margin"] = shape_balance - shape_balance_thr
    out[RULE_RISK_SCORE_COL] = np.minimum(
        np.minimum(
            out["risk_rule_net_load_margin"].astype(float),
            out["risk_rule_hist_centered_margin"].astype(float),
        ),
        np.minimum(
            out["risk_rule_discharge_spike_margin"].astype(float),
            out["risk_rule_discharge_plateau_margin"].astype(float),
        ),
    )
    shape_pass = pd.Series(True, index=out.index)
    if bool(gate.get("use_discharge_shape_gate", False)):
        shape_pass = (
            (out["risk_rule_discharge_spike_margin"].astype(float) >= 0.0)
            & (out["risk_rule_discharge_plateau_margin"].astype(float) >= 0.0)
            & (out["risk_rule_discharge_shape_balance_margin"].astype(float) >= 0.0)
        )
    out[RISK_PROBA_COL] = (
        (out["risk_rule_net_load_margin"].astype(float) >= 0.0)
        & (out["risk_rule_hist_centered_margin"].astype(float) >= 0.0)
        & shape_pass
    ).astype(float)
    structural_pass = pd.Series(True, index=out.index)
    if bool(gate.get("require_both_windows_moved", False)):
        structural_pass = structural_pass & (
            (out["abs_delta_charge_start"].astype(float) > 0.0)
            & (out["abs_delta_discharge_start"].astype(float) > 0.0)
        )
    if bool(gate.get("block_both_windows_earlier", False)):
        structural_pass = structural_pass & ~(
            (out["delta_charge_start"].astype(float) < 0.0)
            & (out["delta_discharge_start"].astype(float) < 0.0)
        )
    out["risk_rule_structural_pass"] = structural_pass.astype(float)
    out.loc[~structural_pass, RISK_PROBA_COL] = 0.0
    out[f"{RISK_PROBA_COL}_std"] = 0.0
    out[RISK_EXPECTED_DELTA_COL] = out[RULE_RISK_SCORE_COL]
    out.loc[~structural_pass, RISK_EXPECTED_DELTA_COL] = -999.0
    out[f"{RISK_EXPECTED_DELTA_COL}_std"] = 0.0
    return out


def select_daily_replacements(
    scored_windows: pd.DataFrame,
    proba_threshold: float,
    min_expected_delta: float,
    min_margin: float = 0.0,
    max_proba_std: float | None = None,
    risk_proba_threshold: float | None = None,
    min_risk_expected_delta: float | None = None,
    require_baseline_stability: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in scored_windows.groupby("date", sort=True):
        eligible = _stage1_eligible_rows(
            group,
            proba_threshold=proba_threshold,
            min_expected_delta=min_expected_delta,
            min_margin=min_margin,
            max_proba_std=max_proba_std,
            require_baseline_stability=require_baseline_stability,
        )

        if eligible.empty:
            first = group.iloc[0]
            rows.append(
                {
                    "date": date,
                    "proposed": False,
                    "selected_true_delta_profit": 0.0,
                    "selected_true_window_profit": float(first.get("baseline_true_window_profit", np.nan)),
                    "baseline_true_window_profit": float(first.get("baseline_true_window_profit", np.nan)),
                    "baseline_charge_start": int(first["baseline_charge_start"]),
                    "baseline_discharge_start": int(first["baseline_discharge_start"]),
                    "candidate_charge_start": int(first["baseline_charge_start"]),
                    "candidate_discharge_start": int(first["baseline_discharge_start"]),
                    PRED_PROBA_COL: 0.0,
                    f"{PRED_PROBA_COL}_std": 0.0,
                    PRED_EXPECTED_DELTA_COL: 0.0,
                    "top1_minus_top2_margin": 0.0,
                    RISK_PROBA_COL: 0.0,
                    f"{RISK_PROBA_COL}_std": 0.0,
                    RISK_EXPECTED_DELTA_COL: 0.0,
                }
            )
            continue

        diagnostic_selected = eligible.sort_values(
            [PRED_EXPECTED_DELTA_COL, PRED_PROBA_COL, "top1_minus_top2_margin"],
            ascending=[False, False, False],
        ).iloc[0]

        final_eligible = eligible.copy()
        if risk_proba_threshold is not None:
            final_eligible = final_eligible.loc[
                final_eligible.get(RISK_PROBA_COL, 0.0).astype(float) >= float(risk_proba_threshold)
            ].copy()
        if min_risk_expected_delta is not None:
            final_eligible = final_eligible.loc[
                final_eligible.get(RISK_EXPECTED_DELTA_COL, 0.0).astype(float)
                >= float(min_risk_expected_delta)
            ].copy()
        if final_eligible.empty:
            rows.append(
                {
                    "date": date,
                    "proposed": False,
                    "selected_true_delta_profit": 0.0,
                    "selected_true_window_profit": float(
                        diagnostic_selected.get("baseline_true_window_profit", np.nan)
                    ),
                    "baseline_true_window_profit": float(
                        diagnostic_selected.get("baseline_true_window_profit", np.nan)
                    ),
                    "baseline_charge_start": int(diagnostic_selected["baseline_charge_start"]),
                    "baseline_discharge_start": int(diagnostic_selected["baseline_discharge_start"]),
                    "candidate_charge_start": int(diagnostic_selected["baseline_charge_start"]),
                    "candidate_discharge_start": int(diagnostic_selected["baseline_discharge_start"]),
                    PRED_PROBA_COL: float(diagnostic_selected[PRED_PROBA_COL]),
                    f"{PRED_PROBA_COL}_std": float(diagnostic_selected[f"{PRED_PROBA_COL}_std"]),
                    PRED_EXPECTED_DELTA_COL: float(diagnostic_selected[PRED_EXPECTED_DELTA_COL]),
                    "top1_minus_top2_margin": float(diagnostic_selected["top1_minus_top2_margin"]),
                    RISK_PROBA_COL: float(diagnostic_selected.get(RISK_PROBA_COL, 0.0)),
                    f"{RISK_PROBA_COL}_std": float(
                        diagnostic_selected.get(f"{RISK_PROBA_COL}_std", 0.0)
                    ),
                    RISK_EXPECTED_DELTA_COL: float(
                        diagnostic_selected.get(RISK_EXPECTED_DELTA_COL, 0.0)
                    ),
                }
            )
            continue
        selected = final_eligible.sort_values(
            [RISK_EXPECTED_DELTA_COL, PRED_EXPECTED_DELTA_COL, PRED_PROBA_COL, "top1_minus_top2_margin"],
            ascending=[False, False, False, False],
        ).iloc[0]
        rows.append(
            {
                "date": date,
                "proposed": True,
                "selected_true_delta_profit": float(selected.get("true_delta_profit", np.nan)),
                "selected_true_window_profit": float(selected.get("true_window_profit", np.nan)),
                "baseline_true_window_profit": float(selected.get("baseline_true_window_profit", np.nan)),
                "baseline_charge_start": int(selected["baseline_charge_start"]),
                "baseline_discharge_start": int(selected["baseline_discharge_start"]),
                "candidate_charge_start": int(selected["charge_start"]),
                "candidate_discharge_start": int(selected["discharge_start"]),
                PRED_PROBA_COL: float(selected[PRED_PROBA_COL]),
                f"{PRED_PROBA_COL}_std": float(selected[f"{PRED_PROBA_COL}_std"]),
                PRED_EXPECTED_DELTA_COL: float(selected[PRED_EXPECTED_DELTA_COL]),
                "top1_minus_top2_margin": float(selected["top1_minus_top2_margin"]),
                RISK_PROBA_COL: float(selected.get(RISK_PROBA_COL, 0.0)),
                f"{RISK_PROBA_COL}_std": float(selected.get(f"{RISK_PROBA_COL}_std", 0.0)),
                RISK_EXPECTED_DELTA_COL: float(selected.get(RISK_EXPECTED_DELTA_COL, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_replacement_metrics(day_metrics: pd.DataFrame) -> dict[str, Any]:
    proposed = day_metrics.loc[day_metrics["proposed"].astype(bool)].copy()
    proposed_days = int(len(proposed))
    false_positive_days = int((proposed["selected_true_delta_profit"].astype(float) <= 0.0).sum())
    positive_days = int((proposed["selected_true_delta_profit"].astype(float) > 0.0).sum())
    return {
        "days": int(len(day_metrics)),
        "proposed_days": proposed_days,
        "positive_selected_days": positive_days,
        "false_positive_days": false_positive_days,
        "false_positive_rate": float(false_positive_days / proposed_days) if proposed_days else 0.0,
        "positive_selected_rate": float(positive_days / proposed_days) if proposed_days else 0.0,
        "avg_delta_all_days": float(day_metrics["selected_true_delta_profit"].mean())
        if len(day_metrics)
        else 0.0,
        "avg_delta_proposed_days": float(proposed["selected_true_delta_profit"].mean())
        if proposed_days
        else 0.0,
        "total_delta_profit": float(day_metrics["selected_true_delta_profit"].sum()),
        "worst_selected_delta": float(proposed["selected_true_delta_profit"].min())
        if proposed_days
        else 0.0,
        "avg_pred_positive_proba": float(proposed[PRED_PROBA_COL].mean()) if proposed_days else 0.0,
        "avg_pred_expected_delta": float(proposed[PRED_EXPECTED_DELTA_COL].mean())
        if proposed_days
        else 0.0,
        "avg_risk_positive_proba": float(proposed[RISK_PROBA_COL].mean())
        if proposed_days and RISK_PROBA_COL in proposed.columns
        else 0.0,
        "avg_risk_expected_delta": float(proposed[RISK_EXPECTED_DELTA_COL].mean())
        if proposed_days and RISK_EXPECTED_DELTA_COL in proposed.columns
        else 0.0,
    }
