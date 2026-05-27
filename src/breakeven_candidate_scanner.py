from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .offline_policy_improvement import (
    add_action_value_features,
    add_submission_price_features,
    baseline_windows_from_submission_frame,
    generate_nearby_actions,
)
from .residual_scenario_generator import DEFAULT_SCENARIO_PREFIX, scenario_columns
from .storage_optimizer import infer_price_column
from .stochastic_candidate_pool import apply_single_day_action


def add_residual_scenario_features(
    actions: pd.DataFrame,
    residual_scenario_df: pd.DataFrame,
    scenario_cols: Optional[Sequence[str]] = None,
    block_size: int = 8,
    power_value: float = 1000.0,
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    if "times" not in residual_scenario_df.columns:
        raise ValueError("residual scenario data missing required column: times")
    cols = list(scenario_cols or scenario_columns(residual_scenario_df))
    if not cols:
        raise ValueError("residual scenario data has no scenario columns")
    missing = [col for col in cols if col not in residual_scenario_df.columns]
    if missing:
        raise ValueError(f"missing residual scenario columns: {missing}")

    df = residual_scenario_df.copy()
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)
    by_date = {date: group.sort_values("times").reset_index(drop=True) for date, group in df.groupby("date")}

    feature_rows: list[dict[str, float]] = []
    for _, action in actions.iterrows():
        date = str(action["date"])
        if date not in by_date:
            raise ValueError(f"residual scenario data missing date: {date}")
        day = by_date[date]
        if len(day) != 96:
            raise ValueError(f"{date} must contain 96 rows, got {len(day)}")
        base_charge = int(action["baseline_charge_start"])
        base_discharge = int(action["baseline_discharge_start"])
        candidate_charge = int(action["candidate_charge_start"])
        candidate_discharge = int(action["candidate_discharge_start"])

        deltas = []
        for col in cols:
            prices = day[col]
            baseline_profit = _window_profit(
                prices,
                base_charge,
                base_discharge,
                block_size=block_size,
                power_value=power_value,
            )
            candidate_profit = _window_profit(
                prices,
                candidate_charge,
                candidate_discharge,
                block_size=block_size,
                power_value=power_value,
            )
            deltas.append(candidate_profit - baseline_profit)
        arr = np.asarray(deltas, dtype=float)
        feature_rows.append(
            {
                "residual_delta_mean": float(np.mean(arr)),
                "residual_delta_std": float(np.std(arr)),
                "residual_delta_min": float(np.min(arr)),
                "residual_delta_p10": float(np.quantile(arr, 0.10)),
                "residual_positive_rate": float((arr > 0.0).mean()),
                "residual_scenario_count": float(len(cols)),
            }
        )
    return pd.concat([actions.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)


def build_breakeven_candidate_pool(
    reference_submission: pd.DataFrame,
    seed_price_df: pd.DataFrame,
    residual_scenario_df: pd.DataFrame,
    max_shift: int = 16,
    blocked_dates: Optional[set[str]] = None,
    min_submission_price_delta: float = 0.0,
    min_pred_seed_delta: float = 1.0,
    min_residual_p10_delta: float = 0.0,
    min_residual_positive_rate: float = 1.0,
    max_selected_abs_start_delta: Optional[int] = 1,
    selection_order: str = "safest",
) -> pd.DataFrame:
    baseline = baseline_windows_from_submission_frame(reference_submission)
    actions = generate_nearby_actions(
        baseline,
        max_shift=int(max_shift),
        blocked_dates=blocked_dates or set(),
    )
    actions = add_action_value_features(actions, seed_price_df)
    price_col = infer_price_column(reference_submission)
    actions = add_submission_price_features(actions, reference_submission, price_col=price_col)
    pre_filter = (
        (actions["submission_price_delta"].astype(float) >= float(min_submission_price_delta))
        & (actions["pred_seed_delta_min"].astype(float) >= float(min_pred_seed_delta))
    )
    if max_selected_abs_start_delta is not None:
        pre_filter &= actions["max_abs_start_delta"].astype(float) <= float(max_selected_abs_start_delta)
    actions = actions.loc[pre_filter].copy()
    if actions.empty:
        return actions

    actions = add_residual_scenario_features(actions, residual_scenario_df)

    actions["multi_price_delta_min"] = np.minimum.reduce(
        [
            actions["submission_price_delta"].astype(float).to_numpy(),
            actions["pred_seed_delta_min"].astype(float).to_numpy(),
            actions["residual_delta_p10"].astype(float).to_numpy(),
        ]
    )
    actions["multi_price_delta_agree"] = (
        (actions["submission_price_delta"].astype(float) >= float(min_submission_price_delta))
        & (actions["pred_seed_delta_min"].astype(float) >= float(min_pred_seed_delta))
        & (actions["residual_delta_p10"].astype(float) >= float(min_residual_p10_delta))
        & (actions["residual_positive_rate"].astype(float) >= float(min_residual_positive_rate))
    )

    filtered = actions.loc[actions["multi_price_delta_agree"]].copy()
    if max_selected_abs_start_delta is not None:
        filtered = filtered.loc[
            filtered["max_abs_start_delta"].astype(float) <= float(max_selected_abs_start_delta)
        ].copy()
    if selection_order == "safest":
        sort_cols = [
            "max_abs_start_delta",
            "submission_price_delta",
            "residual_delta_p10",
            "pred_seed_delta_min",
            "total_abs_start_delta",
        ]
        ascending = [True, False, False, False, True]
    elif selection_order == "strongest":
        sort_cols = [
            "residual_delta_p10",
            "pred_seed_delta_min",
            "max_abs_start_delta",
            "submission_price_delta",
        ]
        ascending = [False, False, True, False]
    else:
        raise ValueError(f"unknown selection_order: {selection_order}")
    return filtered.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def save_breakeven_candidate(
    reference_submission: pd.DataFrame,
    selected: pd.Series,
    output_path: str | Path,
    manifest_path: str | Path,
    reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = apply_single_day_action(reference_submission, selected)
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
                "pred_window_score": float(selected["residual_delta_mean"]),
                "baseline_window_score": 0.0,
                "pred_delta_score": float(selected["residual_delta_p10"]),
                "expected_delta_profit": float(selected["pred_seed_delta_mean"]),
                "score_std": float(selected["residual_delta_std"]),
                "top1_top2_margin": float(selected["pred_seed_delta_min"]),
                "submission_price_delta": float(selected["submission_price_delta"]),
                "multi_price_delta_min": float(selected["multi_price_delta_min"]),
                "multi_price_delta_agree": bool(selected["multi_price_delta_agree"]),
                "pred_seed_delta_min": float(selected["pred_seed_delta_min"]),
                "pred_seed_delta_mean": float(selected["pred_seed_delta_mean"]),
                "pred_seed_delta_positive_count": float(selected["pred_seed_delta_positive_count"]),
                "residual_delta_mean": float(selected["residual_delta_mean"]),
                "residual_delta_p10": float(selected["residual_delta_p10"]),
                "residual_delta_min": float(selected["residual_delta_min"]),
                "residual_positive_rate": float(selected["residual_positive_rate"]),
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


def _parse_blocked_dates(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def _sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a guarded break-even single-day candidate.")
    parser.add_argument("--reference-submission", required=True)
    parser.add_argument("--seed-price-csv", required=True)
    parser.add_argument("--residual-scenario-csv", required=True)
    parser.add_argument("--output", default="outputs/output_breakeven_candidate.csv")
    parser.add_argument("--pool-output", default="outputs/breakeven_candidate_pool.csv")
    parser.add_argument("--manifest-output", default="outputs/breakeven_candidate_manifest.csv")
    parser.add_argument("--max-shift", type=int, default=16)
    parser.add_argument("--max-selected-abs-start-delta", type=int, default=1)
    parser.add_argument("--min-submission-price-delta", type=float, default=0.0)
    parser.add_argument("--min-pred-seed-delta", type=float, default=1.0)
    parser.add_argument("--min-residual-p10-delta", type=float, default=0.0)
    parser.add_argument("--min-residual-positive-rate", type=float, default=1.0)
    parser.add_argument("--selection-order", choices=["safest", "strongest"], default="safest")
    parser.add_argument("--blocked-dates", default="")
    parser.add_argument("--reason", default="break-even submission-price candidate with positive seed and residual scenarios")
    args = parser.parse_args()

    reference = pd.read_csv(args.reference_submission)
    seed_price = pd.read_csv(args.seed_price_csv)
    residual_scenarios = pd.read_csv(args.residual_scenario_csv)
    selected_max_delta = (
        None if int(args.max_selected_abs_start_delta) < 0 else int(args.max_selected_abs_start_delta)
    )
    pool = build_breakeven_candidate_pool(
        reference,
        seed_price,
        residual_scenarios,
        max_shift=args.max_shift,
        blocked_dates=_parse_blocked_dates(args.blocked_dates),
        min_submission_price_delta=args.min_submission_price_delta,
        min_pred_seed_delta=args.min_pred_seed_delta,
        min_residual_p10_delta=args.min_residual_p10_delta,
        min_residual_positive_rate=args.min_residual_positive_rate,
        max_selected_abs_start_delta=selected_max_delta,
        selection_order=args.selection_order,
    )
    if pool.empty:
        raise SystemExit("no breakeven candidate passed the configured gates")

    pool.insert(0, "pool_rank", np.arange(1, len(pool) + 1))
    pool_path = Path(args.pool_output)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(pool_path, index=False)

    selected = pool.iloc[0]
    save_breakeven_candidate(reference, selected, args.output, args.manifest_output, args.reason)
    candidate_sha = _sha256(args.output)
    print(
        "selected_breakeven_candidate="
        f"date={selected['date']}, baseline={int(selected['baseline_charge_start'])}/"
        f"{int(selected['baseline_discharge_start'])}, candidate="
        f"{int(selected['candidate_charge_start'])}/{int(selected['candidate_discharge_start'])}, "
        f"submission_price_delta={float(selected['submission_price_delta']):.6f}, "
        f"pred_seed_delta_min={float(selected['pred_seed_delta_min']):.6f}, "
        f"residual_delta_p10={float(selected['residual_delta_p10']):.6f}"
    )
    print(
        f"saved_submission={args.output}, pool={args.pool_output}, "
        f"manifest={args.manifest_output}, sha256={candidate_sha}"
    )


if __name__ == "__main__":
    main()
