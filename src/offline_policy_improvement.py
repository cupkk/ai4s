from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .stochastic_optimizer import optimize_one_day_scenarios


DEFAULT_SEED_PREFIX = "pred_price_seed"


def baseline_windows_from_submission_frame(submission: pd.DataFrame) -> pd.DataFrame:
    if "times" not in submission.columns or "power" not in submission.columns:
        raise ValueError("submission must contain times and power columns")
    df = submission.copy()
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)
    rows: list[dict[str, int | str]] = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        power = group["power"].to_numpy(dtype=float)
        charge = np.flatnonzero(power < 0)
        discharge = np.flatnonzero(power > 0)
        if len(charge) == 0 or len(discharge) == 0:
            continue
        rows.append(
            {
                "date": str(date),
                "baseline_charge_start": int(charge[0]),
                "baseline_discharge_start": int(discharge[0]),
            }
        )
    return pd.DataFrame(rows)


def baseline_windows_from_submission(path: str | Path) -> pd.DataFrame:
    return baseline_windows_from_submission_frame(pd.read_csv(path))


def baseline_windows_from_prediction_policy(
    price_df: pd.DataFrame,
    seed_cols: Sequence[str],
    risk_lambda: float = 0.5,
) -> pd.DataFrame:
    df = price_df.copy()
    if "times" not in df.columns:
        raise ValueError("price_df missing required column: times")
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)
    missing = [col for col in seed_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing seed columns: {missing}")

    rows: list[dict[str, int | str]] = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        if len(group) != 96:
            continue
        result = optimize_one_day_scenarios(
            group[list(seed_cols)].to_numpy(dtype=float).T,
            risk_lambda=float(risk_lambda),
        )
        if result.traded and result.charge_start is not None and result.discharge_start is not None:
            rows.append(
                {
                    "date": str(date),
                    "baseline_charge_start": int(result.charge_start),
                    "baseline_discharge_start": int(result.discharge_start),
                }
            )
    if not rows:
        raise ValueError("prediction policy produced no complete-day baseline windows")
    return pd.DataFrame(rows)


def generate_nearby_actions(
    baseline_windows: pd.DataFrame,
    max_shift: int,
    block_size: int = 8,
    slots_per_day: int = 96,
    blocked_dates: Optional[set[str]] = None,
) -> pd.DataFrame:
    required = {"date", "baseline_charge_start", "baseline_discharge_start"}
    missing = required.difference(baseline_windows.columns)
    if missing:
        raise ValueError(f"baseline_windows missing required columns: {sorted(missing)}")

    blocked_dates = blocked_dates or set()
    rows: list[dict[str, int | str]] = []
    max_charge_start = slots_per_day - 2 * block_size
    max_start = slots_per_day - block_size
    for _, row in baseline_windows.iterrows():
        date = str(row["date"])
        if date in blocked_dates:
            continue
        base_charge = int(row["baseline_charge_start"])
        base_discharge = int(row["baseline_discharge_start"])
        for charge in range(base_charge - int(max_shift), base_charge + int(max_shift) + 1):
            if charge < 0 or charge > max_charge_start:
                continue
            for discharge in range(
                base_discharge - int(max_shift),
                base_discharge + int(max_shift) + 1,
            ):
                if discharge < charge + block_size or discharge > max_start:
                    continue
                if charge == base_charge and discharge == base_discharge:
                    continue
                delta_charge = charge - base_charge
                delta_discharge = discharge - base_discharge
                rows.append(
                    {
                        "date": date,
                        "baseline_charge_start": base_charge,
                        "baseline_discharge_start": base_discharge,
                        "candidate_charge_start": int(charge),
                        "candidate_discharge_start": int(discharge),
                        "delta_charge_start": int(delta_charge),
                        "delta_discharge_start": int(delta_discharge),
                        "abs_delta_charge_start": abs(int(delta_charge)),
                        "abs_delta_discharge_start": abs(int(delta_discharge)),
                        "max_abs_start_delta": max(abs(int(delta_charge)), abs(int(delta_discharge))),
                        "total_abs_start_delta": abs(int(delta_charge)) + abs(int(delta_discharge)),
                        "baseline_gap_slots": base_discharge - base_charge,
                        "candidate_gap_slots": int(discharge) - int(charge),
                        "delta_gap_slots": int(discharge) - int(charge) - (base_discharge - base_charge),
                    }
                )
    return pd.DataFrame(rows)


def add_action_value_features(
    actions: pd.DataFrame,
    price_df: pd.DataFrame,
    seed_cols: Optional[Sequence[str]] = None,
    true_col: str = "",
    block_size: int = 8,
    power_value: float = 1000.0,
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    df = price_df.copy()
    if "times" not in df.columns:
        raise ValueError("price_df missing required column: times")
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)
    cols = list(seed_cols or [col for col in df.columns if col.startswith(DEFAULT_SEED_PREFIX)])
    if not cols:
        raise ValueError("no seed prediction columns available")
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing seed columns: {missing}")
    if true_col and true_col not in df.columns:
        raise ValueError(f"true_col not found: {true_col}")
    by_date = {date: group.sort_values("times").reset_index(drop=True) for date, group in df.groupby("date")}

    feature_rows: list[dict[str, float]] = []
    for _, action in actions.iterrows():
        date = str(action["date"])
        if date not in by_date:
            raise ValueError(f"price_df missing date: {date}")
        day = by_date[date]
        if len(day) != 96:
            raise ValueError(f"{date} must contain 96 rows, got {len(day)}")
        base_charge = int(action["baseline_charge_start"])
        base_discharge = int(action["baseline_discharge_start"])
        cand_charge = int(action["candidate_charge_start"])
        cand_discharge = int(action["candidate_discharge_start"])

        row_features: dict[str, float] = {}
        seed_deltas: list[float] = []
        seed_candidate_profits: list[float] = []
        seed_baseline_profits: list[float] = []
        for col in cols:
            baseline_profit = _window_profit(
                day[col],
                base_charge,
                base_discharge,
                block_size=block_size,
                power_value=power_value,
            )
            candidate_profit = _window_profit(
                day[col],
                cand_charge,
                cand_discharge,
                block_size=block_size,
                power_value=power_value,
            )
            delta = candidate_profit - baseline_profit
            row_features[f"{col}_candidate_profit"] = candidate_profit
            row_features[f"{col}_baseline_profit"] = baseline_profit
            row_features[f"{col}_delta"] = delta
            seed_deltas.append(delta)
            seed_candidate_profits.append(candidate_profit)
            seed_baseline_profits.append(baseline_profit)

        row_features["pred_seed_delta_min"] = float(np.min(seed_deltas))
        row_features["pred_seed_delta_mean"] = float(np.mean(seed_deltas))
        row_features["pred_seed_delta_std"] = float(np.std(seed_deltas))
        row_features["pred_seed_delta_positive_count"] = float(sum(delta > 0 for delta in seed_deltas))
        row_features["pred_seed_candidate_profit_mean"] = float(np.mean(seed_candidate_profits))
        row_features["pred_seed_baseline_profit_mean"] = float(np.mean(seed_baseline_profits))
        row_features["pred_seed_candidate_profit_std"] = float(np.std(seed_candidate_profits))
        row_features["pred_seed_baseline_profit_std"] = float(np.std(seed_baseline_profits))

        if true_col:
            true_baseline = _window_profit(
                day[true_col],
                base_charge,
                base_discharge,
                block_size=block_size,
                power_value=power_value,
            )
            true_candidate = _window_profit(
                day[true_col],
                cand_charge,
                cand_discharge,
                block_size=block_size,
                power_value=power_value,
            )
            row_features["true_baseline_profit"] = true_baseline
            row_features["true_candidate_profit"] = true_candidate
            row_features["true_delta_profit"] = true_candidate - true_baseline
        feature_rows.append(row_features)

    return pd.concat([actions.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)


def add_submission_price_features(
    actions: pd.DataFrame,
    reference_submission: pd.DataFrame,
    price_col: str = "实时价格",
    block_size: int = 8,
    power_value: float = 1000.0,
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    if "times" not in reference_submission.columns:
        raise ValueError("reference_submission missing required column: times")
    if price_col not in reference_submission.columns:
        raise ValueError(f"reference_submission missing price column: {price_col}")

    ref = reference_submission.copy()
    ref["times"] = pd.to_datetime(ref["times"])
    ref["date"] = ref["times"].dt.date.astype(str)
    by_date = {date: group.sort_values("times").reset_index(drop=True) for date, group in ref.groupby("date")}

    feature_rows: list[dict[str, float]] = []
    for _, action in actions.iterrows():
        date = str(action["date"])
        if date not in by_date:
            raise ValueError(f"reference_submission missing date: {date}")
        day = by_date[date]
        if len(day) != 96:
            raise ValueError(f"{date} must contain 96 rows, got {len(day)}")
        base_charge = int(action["baseline_charge_start"])
        base_discharge = int(action["baseline_discharge_start"])
        cand_charge = int(action["candidate_charge_start"])
        cand_discharge = int(action["candidate_discharge_start"])
        baseline_profit = _window_profit(
            day[price_col],
            base_charge,
            base_discharge,
            block_size=block_size,
            power_value=power_value,
        )
        candidate_profit = _window_profit(
            day[price_col],
            cand_charge,
            cand_discharge,
            block_size=block_size,
            power_value=power_value,
        )
        delta = candidate_profit - baseline_profit
        feature_rows.append(
            {
                "submission_price_baseline_profit": baseline_profit,
                "submission_price_candidate_profit": candidate_profit,
                "submission_price_delta": delta,
                "multi_price_delta_min": min(float(delta), float(action.get("pred_seed_delta_min", delta))),
                "multi_price_delta_agree": bool(delta > 0 and float(action.get("pred_seed_delta_min", 0.0)) > 0),
            }
        )
    return pd.concat([actions.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)


def add_shape_risk_features(
    candidates: pd.DataFrame,
    historical_actions: pd.DataFrame,
    quantile: float = 0.10,
    group_cols: Sequence[str] = ("delta_charge_start", "delta_discharge_start"),
    label_col: str = "true_delta_profit",
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    missing_candidates = set(group_cols).difference(candidates.columns)
    if missing_candidates:
        raise ValueError(f"candidates missing shape columns: {sorted(missing_candidates)}")
    required_history = set(group_cols).union({label_col})
    missing_history = required_history.difference(historical_actions.columns)
    if missing_history:
        raise ValueError(f"historical_actions missing shape-risk columns: {sorted(missing_history)}")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")

    hist = historical_actions[list(group_cols) + [label_col]].copy()
    hist[label_col] = hist[label_col].astype(float)
    grouped = hist.groupby(list(group_cols), dropna=False)[label_col]
    stats = grouped.agg(
        shape_sample_count="count",
        shape_true_delta_min="min",
        shape_true_delta_mean="mean",
        shape_true_delta_p10=lambda values: float(np.quantile(values.to_numpy(dtype=float), float(quantile))),
        shape_positive_rate=lambda values: float((values.to_numpy(dtype=float) > 0.0).mean()),
    ).reset_index()
    out = candidates.merge(stats, on=list(group_cols), how="left")
    out["shape_sample_count"] = out["shape_sample_count"].fillna(0).astype(int)
    return out


def rank_policy_candidates(
    candidates: pd.DataFrame,
    lower_confidence_lambda: float,
    min_offline_delta_lower: float,
    min_pred_seed_delta: float,
    min_seed_positive_count: int,
    max_abs_start_delta: Optional[int] = None,
    blocked_dates: Optional[set[str]] = None,
    min_delta_gap_slots: Optional[int] = None,
    forbid_charge_later_discharge_earlier: bool = False,
    min_submission_price_delta: Optional[float] = None,
    min_shape_sample_count: Optional[int] = None,
    min_shape_positive_rate: Optional[float] = None,
    min_shape_p10_delta: Optional[float] = None,
) -> pd.DataFrame:
    out = add_policy_gate_columns(
        candidates,
        lower_confidence_lambda=lower_confidence_lambda,
        min_offline_delta_lower=min_offline_delta_lower,
        min_pred_seed_delta=min_pred_seed_delta,
        min_seed_positive_count=min_seed_positive_count,
        max_abs_start_delta=max_abs_start_delta,
        blocked_dates=blocked_dates,
        min_delta_gap_slots=min_delta_gap_slots,
        forbid_charge_later_discharge_earlier=forbid_charge_later_discharge_earlier,
        min_submission_price_delta=min_submission_price_delta,
        min_shape_sample_count=min_shape_sample_count,
        min_shape_positive_rate=min_shape_positive_rate,
        min_shape_p10_delta=min_shape_p10_delta,
    )
    out = out.loc[out["passes_policy_gate"]].copy()
    if out.empty:
        return out
    return out.sort_values(
        [
            "offline_pred_delta_lower",
            "pred_seed_delta_min",
            "offline_pred_delta_mean",
            "total_abs_start_delta",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def add_policy_gate_columns(
    candidates: pd.DataFrame,
    lower_confidence_lambda: float,
    min_offline_delta_lower: float,
    min_pred_seed_delta: float,
    min_seed_positive_count: int,
    max_abs_start_delta: Optional[int] = None,
    blocked_dates: Optional[set[str]] = None,
    min_delta_gap_slots: Optional[int] = None,
    forbid_charge_later_discharge_earlier: bool = False,
    min_submission_price_delta: Optional[float] = None,
    min_shape_sample_count: Optional[int] = None,
    min_shape_positive_rate: Optional[float] = None,
    min_shape_p10_delta: Optional[float] = None,
) -> pd.DataFrame:
    required = {
        "offline_pred_delta_mean",
        "offline_pred_delta_std",
        "pred_seed_delta_min",
        "pred_seed_delta_positive_count",
        "max_abs_start_delta",
        "total_abs_start_delta",
        "date",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing ranking columns: {sorted(missing)}")
    out = candidates.copy()
    out["offline_pred_delta_lower"] = (
        out["offline_pred_delta_mean"].astype(float)
        - float(lower_confidence_lambda) * out["offline_pred_delta_std"].astype(float)
    )
    passes = (
        (out["offline_pred_delta_lower"].astype(float) >= float(min_offline_delta_lower))
        & (out["pred_seed_delta_min"].astype(float) >= float(min_pred_seed_delta))
        & (out["pred_seed_delta_positive_count"].astype(float) >= float(min_seed_positive_count))
    )
    if blocked_dates:
        passes &= ~out["date"].astype(str).isin(blocked_dates)
    if max_abs_start_delta is not None:
        passes &= out["max_abs_start_delta"].astype(float) <= float(max_abs_start_delta)
    if min_delta_gap_slots is not None:
        if "delta_gap_slots" not in out.columns:
            raise ValueError("candidates missing required column for gap gate: delta_gap_slots")
        passes &= out["delta_gap_slots"].astype(float) >= float(min_delta_gap_slots)
    if forbid_charge_later_discharge_earlier:
        missing_shape_cols = {"delta_charge_start", "delta_discharge_start"}.difference(out.columns)
        if missing_shape_cols:
            raise ValueError(
                f"candidates missing required columns for shape gate: {sorted(missing_shape_cols)}"
            )
        compressed_middle = (
            (out["delta_charge_start"].astype(float) > 0)
            & (out["delta_discharge_start"].astype(float) < 0)
        )
        passes &= ~compressed_middle
    if min_submission_price_delta is not None:
        if "submission_price_delta" not in out.columns:
            raise ValueError("candidates missing required column: submission_price_delta")
        passes &= out["submission_price_delta"].astype(float) >= float(min_submission_price_delta)
    if min_shape_sample_count is not None:
        if "shape_sample_count" not in out.columns:
            raise ValueError("candidates missing required column: shape_sample_count")
        passes &= out["shape_sample_count"].astype(float) >= float(min_shape_sample_count)
    if min_shape_positive_rate is not None:
        if "shape_positive_rate" not in out.columns:
            raise ValueError("candidates missing required column: shape_positive_rate")
        passes &= out["shape_positive_rate"].astype(float) >= float(min_shape_positive_rate)
    if min_shape_p10_delta is not None:
        if "shape_true_delta_p10" not in out.columns:
            raise ValueError("candidates missing required column: shape_true_delta_p10")
        passes &= out["shape_true_delta_p10"].astype(float) >= float(min_shape_p10_delta)
    out["passes_policy_gate"] = passes
    return out


def validate_policy_split(
    price_df: pd.DataFrame,
    seed_cols: Sequence[str],
    split_date: str,
    model_seeds: Sequence[int],
    behavior_risk_lambda: float,
    train_max_shift: int,
    validation_max_shift: int,
    lower_confidence_lambda: float,
    min_offline_delta_lower: float,
    min_pred_seed_delta: float,
    min_seed_positive_count: int,
    true_col: str = "A",
    top_k: int = 30,
    min_delta_gap_slots: Optional[int] = None,
    forbid_charge_later_discharge_earlier: bool = False,
) -> pd.DataFrame:
    df = price_df.copy()
    if "times" not in df.columns:
        raise ValueError("price_df missing required column: times")
    df["times"] = pd.to_datetime(df["times"])
    split_ts = pd.Timestamp(split_date)
    train_df = df.loc[df["times"] < split_ts].copy()
    validation_df = df.loc[df["times"] >= split_ts].copy()
    if train_df.empty or validation_df.empty:
        raise ValueError(
            f"split_date={split_date} must leave non-empty train and validation partitions"
        )

    train_baseline = baseline_windows_from_prediction_policy(
        train_df,
        seed_cols=seed_cols,
        risk_lambda=behavior_risk_lambda,
    )
    train_actions = generate_nearby_actions(
        train_baseline,
        max_shift=train_max_shift,
        blocked_dates=set(),
    )
    train_actions = add_action_value_features(
        train_actions,
        train_df,
        seed_cols=seed_cols,
        true_col=true_col,
    )
    feature_columns = default_feature_columns(train_actions)
    models = train_delta_models(
        train_actions,
        feature_columns=feature_columns,
        seeds=model_seeds,
    )

    validation_baseline = baseline_windows_from_prediction_policy(
        validation_df,
        seed_cols=seed_cols,
        risk_lambda=behavior_risk_lambda,
    )
    validation_actions = generate_nearby_actions(
        validation_baseline,
        max_shift=validation_max_shift,
        blocked_dates=set(),
    )
    validation_actions = add_action_value_features(
        validation_actions,
        validation_df,
        seed_cols=seed_cols,
        true_col=true_col,
    )
    scored = add_model_predictions(validation_actions, models=models, feature_columns=feature_columns)
    gated = add_policy_gate_columns(
        scored,
        lower_confidence_lambda=lower_confidence_lambda,
        min_offline_delta_lower=min_offline_delta_lower,
        min_pred_seed_delta=min_pred_seed_delta,
        min_seed_positive_count=min_seed_positive_count,
        max_abs_start_delta=validation_max_shift,
        blocked_dates=set(),
        min_delta_gap_slots=min_delta_gap_slots,
        forbid_charge_later_discharge_earlier=forbid_charge_later_discharge_earlier,
    )
    gated = gated.sort_values(
        [
            "passes_policy_gate",
            "offline_pred_delta_lower",
            "pred_seed_delta_min",
            "offline_pred_delta_mean",
            "total_abs_start_delta",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    gated["policy_rank"] = np.nan
    if not gated.empty:
        passed = gated["passes_policy_gate"].to_numpy(dtype=bool)
        gated.loc[passed, "policy_rank"] = np.arange(1, int(passed.sum()) + 1)
        gated["selected_by_policy"] = gated["policy_rank"].eq(1)
    else:
        gated["selected_by_policy"] = []
    return gated.head(int(top_k)).copy()


def train_delta_models(
    train_actions: pd.DataFrame,
    feature_columns: Sequence[str],
    seeds: Sequence[int],
    label_col: str = "true_delta_profit",
) -> list[object]:
    import lightgbm as lgb

    if label_col not in train_actions.columns:
        raise ValueError(f"train_actions missing label column: {label_col}")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    x = train_actions[list(feature_columns)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_actions[label_col].to_numpy(dtype=float)
    models: list[object] = []
    for seed in seeds:
        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "min_data_in_leaf": 24,
            "seed": int(seed),
            "verbose": -1,
        }
        model = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=160)
        models.append(model)
    return models


def add_model_predictions(
    candidates: pd.DataFrame,
    models: Sequence[object],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    if not models:
        raise ValueError("models must not be empty")
    x = candidates[list(feature_columns)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pred_matrix = np.vstack([model.predict(x) for model in models])
    out = candidates.copy()
    for idx, pred in enumerate(pred_matrix):
        out[f"offline_pred_delta_model{idx}"] = pred
    out["offline_pred_delta_mean"] = pred_matrix.mean(axis=0)
    out["offline_pred_delta_std"] = pred_matrix.std(axis=0)
    return out


def default_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "true_delta_profit",
        "true_baseline_profit",
        "true_candidate_profit",
    }
    numeric_cols = [
        col
        for col in frame.select_dtypes(include=[np.number]).columns
        if col not in excluded
    ]
    return numeric_cols


def save_single_day_candidate(
    reference_submission: pd.DataFrame,
    selected: pd.Series,
    output_path: str | Path,
    manifest_path: str | Path,
    reason: str,
    block_size: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .stochastic_candidate_pool import apply_single_day_action

    submission = apply_single_day_action(
        reference_submission,
        pd.Series(
            {
                "date": selected["date"],
                "candidate_charge_start": int(selected["candidate_charge_start"]),
                "candidate_discharge_start": int(selected["candidate_discharge_start"]),
            }
        ),
        block_size=block_size,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    candidate_sha = _sha256(output_path)
    manifest = pd.DataFrame(
        [
            {
                "candidate_csv": output_path.as_posix(),
                "candidate_sha256": candidate_sha,
                "date": selected["date"],
                "blocked": False,
                "changed_days": 1,
                "baseline_charge_start": int(selected["baseline_charge_start"]),
                "baseline_discharge_start": int(selected["baseline_discharge_start"]),
                "candidate_charge_start": int(selected["candidate_charge_start"]),
                "candidate_discharge_start": int(selected["candidate_discharge_start"]),
                "pred_window_score": float(selected["offline_pred_delta_mean"]),
                "baseline_window_score": 0.0,
                "pred_delta_score": float(selected["offline_pred_delta_lower"]),
                "expected_delta_profit": float(selected["pred_seed_delta_mean"]),
                "score_std": float(selected["offline_pred_delta_std"]),
                "top1_top2_margin": float(selected.get("pred_seed_delta_min", 0.0)),
                "offline_pred_delta_lower": float(selected["offline_pred_delta_lower"]),
                "pred_seed_delta_min": float(selected["pred_seed_delta_min"]),
                "pred_seed_delta_mean": float(selected["pred_seed_delta_mean"]),
                "pred_seed_delta_positive_count": float(selected["pred_seed_delta_positive_count"]),
                "submission_price_delta": float(selected.get("submission_price_delta", np.nan)),
                "multi_price_delta_min": float(selected.get("multi_price_delta_min", np.nan)),
                "multi_price_delta_agree": bool(selected.get("multi_price_delta_agree", False)),
                "shape_sample_count": float(selected.get("shape_sample_count", np.nan)),
                "shape_positive_rate": float(selected.get("shape_positive_rate", np.nan)),
                "shape_true_delta_p10": float(selected.get("shape_true_delta_p10", np.nan)),
                "shape_true_delta_min": float(selected.get("shape_true_delta_min", np.nan)),
                "reason": reason,
            }
        ]
    )
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return submission, manifest


def _window_profit(
    prices: pd.Series,
    charge_start: int,
    discharge_start: int,
    block_size: int,
    power_value: float,
) -> float:
    charge = prices.iloc[charge_start : charge_start + block_size].sum()
    discharge = prices.iloc[discharge_start : discharge_start + block_size].sum()
    return float(power_value * (discharge - charge))


def parse_blocked_dates(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("integer list must not be empty")
    return values


def _sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative offline policy-improvement candidate generator.")
    parser.add_argument("--train-pred-csv", required=True)
    parser.add_argument("--test-pred-csv", required=True)
    parser.add_argument("--reference-submission", required=True)
    parser.add_argument("--output", default="outputs/output_offline_policy_candidate.csv")
    parser.add_argument("--pool-output", default="outputs/offline_policy_candidate_pool.csv")
    parser.add_argument("--manifest-output", default="outputs/offline_policy_manifest.csv")
    parser.add_argument("--true-col", default="A")
    parser.add_argument("--seed-cols", default="")
    parser.add_argument("--model-seeds", default="42,2024,2026")
    parser.add_argument("--behavior-risk-lambda", type=float, default=0.5)
    parser.add_argument("--train-max-shift", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--blocked-dates", default="")
    parser.add_argument("--lower-confidence-lambda", type=float, default=1.0)
    parser.add_argument("--min-offline-delta-lower", type=float, default=100.0)
    parser.add_argument("--min-pred-seed-delta", type=float, default=100.0)
    parser.add_argument("--min-seed-positive-count", type=int, default=3)
    parser.add_argument("--submission-price-col", default="实时价格")
    parser.add_argument(
        "--min-submission-price-delta",
        type=float,
        default=0.0,
        help="Minimum delta under the reference submission price column. Use a negative value only for diagnostics.",
    )
    parser.add_argument("--shape-risk-quantile", type=float, default=0.10)
    parser.add_argument("--min-shape-sample-count", type=int, default=0)
    parser.add_argument("--min-shape-positive-rate", type=float, default=0.0)
    parser.add_argument("--min-shape-p10-delta", type=float, default=-1.0e18)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Write the candidate pool only; do not create a submission or manifest.",
    )
    parser.add_argument(
        "--min-delta-gap-slots",
        type=int,
        default=None,
        help="Optional action-shape gate: candidate gap change must be at least this value.",
    )
    parser.add_argument(
        "--forbid-charge-later-discharge-earlier",
        action="store_true",
        help="Reject actions that move charge later and discharge earlier at the same time.",
    )
    parser.add_argument(
        "--validation-split-date",
        default="",
        help="Optional historical holdout start date, e.g. 2025-02-01.",
    )
    parser.add_argument(
        "--validation-output",
        default="",
        help="Optional CSV path for historical holdout policy validation details.",
    )
    parser.add_argument(
        "--min-validation-selected-delta",
        type=float,
        default=0.0,
        help="Abort candidate generation if the holdout top policy action is below this true delta.",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--reason", default="conservative offline policy-improvement single-day candidate")
    args = parser.parse_args()

    train_pred = pd.read_csv(args.train_pred_csv)
    test_pred = pd.read_csv(args.test_pred_csv)
    reference = pd.read_csv(args.reference_submission)
    seed_cols = (
        [item.strip() for item in args.seed_cols.split(",") if item.strip()]
        if args.seed_cols
        else [col for col in test_pred.columns if col.startswith(DEFAULT_SEED_PREFIX)]
    )
    seed_cols = [col for col in seed_cols if col in train_pred.columns and col in test_pred.columns]
    if not seed_cols:
        raise ValueError("no shared seed prediction columns found between train and test")

    if args.validation_split_date:
        validation = validate_policy_split(
            train_pred,
            seed_cols=seed_cols,
            split_date=args.validation_split_date,
            model_seeds=parse_ints(args.model_seeds),
            behavior_risk_lambda=args.behavior_risk_lambda,
            train_max_shift=args.train_max_shift,
            validation_max_shift=args.max_shift,
            lower_confidence_lambda=args.lower_confidence_lambda,
            min_offline_delta_lower=args.min_offline_delta_lower,
            min_pred_seed_delta=args.min_pred_seed_delta,
            min_seed_positive_count=args.min_seed_positive_count,
            true_col=args.true_col,
            top_k=args.top_k,
            min_delta_gap_slots=args.min_delta_gap_slots,
            forbid_charge_later_discharge_earlier=args.forbid_charge_later_discharge_earlier,
        )
        if args.validation_output:
            validation_path = Path(args.validation_output)
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            validation.to_csv(validation_path, index=False)
        selected_validation = validation.loc[validation["selected_by_policy"].astype(bool)]
        if selected_validation.empty:
            print(
                "offline_policy_validation=none, "
                f"split_date={args.validation_split_date}, "
                f"rows={len(validation)}, output={args.validation_output or ''}"
            )
            return
        validation_top = selected_validation.iloc[0]
        validation_delta = float(validation_top["true_delta_profit"])
        print(
            "offline_policy_validation="
            f"date={validation_top['date']}, "
            f"baseline={int(validation_top['baseline_charge_start'])}/{int(validation_top['baseline_discharge_start'])}, "
            f"candidate={int(validation_top['candidate_charge_start'])}/{int(validation_top['candidate_discharge_start'])}, "
            f"true_delta={validation_delta:.6f}, "
            f"offline_lower={float(validation_top['offline_pred_delta_lower']):.6f}"
        )
        if validation_delta < float(args.min_validation_selected_delta):
            print(
                "offline_policy_candidate=aborted_by_validation, "
                f"true_delta={validation_delta:.6f}, "
                f"min_validation_selected_delta={float(args.min_validation_selected_delta):.6f}"
            )
            return

    train_baseline = baseline_windows_from_prediction_policy(
        train_pred,
        seed_cols=seed_cols,
        risk_lambda=args.behavior_risk_lambda,
    )
    train_actions = generate_nearby_actions(
        train_baseline,
        max_shift=args.train_max_shift,
        blocked_dates=set(),
    )
    train_actions = add_action_value_features(
        train_actions,
        train_pred,
        seed_cols=seed_cols,
        true_col=args.true_col,
    )
    feature_columns = default_feature_columns(train_actions)
    models = train_delta_models(
        train_actions,
        feature_columns=feature_columns,
        seeds=parse_ints(args.model_seeds),
    )

    test_baseline = baseline_windows_from_submission_frame(reference)
    test_actions = generate_nearby_actions(
        test_baseline,
        max_shift=args.max_shift,
        blocked_dates=parse_blocked_dates(args.blocked_dates),
    )
    test_actions = add_action_value_features(
        test_actions,
        test_pred,
        seed_cols=seed_cols,
        true_col="",
    )
    if args.submission_price_col:
        test_actions = add_submission_price_features(
            test_actions,
            reference,
            price_col=args.submission_price_col,
        )
    scored = add_model_predictions(test_actions, models=models, feature_columns=feature_columns)
    scored = add_shape_risk_features(
        scored,
        train_actions,
        quantile=args.shape_risk_quantile,
    )
    ranked = rank_policy_candidates(
        scored,
        lower_confidence_lambda=args.lower_confidence_lambda,
        min_offline_delta_lower=args.min_offline_delta_lower,
        min_pred_seed_delta=args.min_pred_seed_delta,
        min_seed_positive_count=args.min_seed_positive_count,
        max_abs_start_delta=args.max_shift,
        blocked_dates=parse_blocked_dates(args.blocked_dates),
        min_delta_gap_slots=args.min_delta_gap_slots,
        forbid_charge_later_discharge_earlier=args.forbid_charge_later_discharge_earlier,
        min_submission_price_delta=(
            args.min_submission_price_delta if args.submission_price_col else None
        ),
        min_shape_sample_count=args.min_shape_sample_count,
        min_shape_positive_rate=args.min_shape_positive_rate,
        min_shape_p10_delta=args.min_shape_p10_delta,
    )
    Path(args.pool_output).parent.mkdir(parents=True, exist_ok=True)
    pool_to_save = ranked.head(int(args.top_k)).copy() if not ranked.empty else add_policy_gate_columns(
        scored,
        lower_confidence_lambda=args.lower_confidence_lambda,
        min_offline_delta_lower=args.min_offline_delta_lower,
        min_pred_seed_delta=args.min_pred_seed_delta,
        min_seed_positive_count=args.min_seed_positive_count,
        max_abs_start_delta=args.max_shift,
        blocked_dates=parse_blocked_dates(args.blocked_dates),
        min_delta_gap_slots=args.min_delta_gap_slots,
        forbid_charge_later_discharge_earlier=args.forbid_charge_later_discharge_earlier,
        min_submission_price_delta=(
            args.min_submission_price_delta if args.submission_price_col else None
        ),
        min_shape_sample_count=args.min_shape_sample_count,
        min_shape_positive_rate=args.min_shape_positive_rate,
        min_shape_p10_delta=args.min_shape_p10_delta,
    ).sort_values(
        [
            "passes_policy_gate",
            "offline_pred_delta_lower",
            "pred_seed_delta_min",
            "submission_price_delta" if args.submission_price_col else "offline_pred_delta_mean",
            "total_abs_start_delta",
        ],
        ascending=[False, False, False, False, True],
    ).head(int(args.top_k)).copy()
    pool_to_save.to_csv(args.pool_output, index=False)
    if ranked.empty:
        print(
            "offline_policy_candidate=none, "
            f"train_rows={len(train_actions)}, test_rows={len(test_actions)}, "
            f"pool={args.pool_output}"
        )
        return
    if args.diagnostic_only:
        print(
            "offline_policy_candidate=diagnostic_only, "
            f"ranked_rows={len(ranked)}, pool={args.pool_output}"
        )
        return

    selected = ranked.iloc[0]
    save_single_day_candidate(
        reference,
        selected,
        output_path=args.output,
        manifest_path=args.manifest_output,
        reason=args.reason,
    )
    print(
        "offline_policy_candidate="
        f"date={selected['date']}, "
        f"baseline={int(selected['baseline_charge_start'])}/{int(selected['baseline_discharge_start'])}, "
        f"candidate={int(selected['candidate_charge_start'])}/{int(selected['candidate_discharge_start'])}, "
        f"offline_lower={float(selected['offline_pred_delta_lower']):.6f}, "
        f"pred_seed_min={float(selected['pred_seed_delta_min']):.6f}, "
        f"sha256={_sha256(args.output)}"
    )


if __name__ == "__main__":
    main()
