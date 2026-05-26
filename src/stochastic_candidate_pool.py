from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .offline_policy_improvement import add_submission_price_features
from .residual_scenario_generator import DEFAULT_SCENARIO_PREFIX
from .storage_optimizer import infer_price_column
from .stochastic_optimizer import generate_stochastic_strategy


ScenarioSet = tuple[str, list[str]]


def parse_float_grid(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("float grid must contain at least one value")
    return values


def parse_int_grid(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("integer grid must contain at least one value")
    return values


def parse_blocked_dates(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def default_scenario_sets(df: pd.DataFrame) -> list[ScenarioSet]:
    residual_cols = sorted(col for col in df.columns if col.startswith(DEFAULT_SCENARIO_PREFIX))
    if len(residual_cols) >= 2:
        return [("residual_day_resample", residual_cols)]

    seed_cols = [col for col in df.columns if col.startswith("pred_price_seed")]
    if len(seed_cols) >= 2:
        out: list[ScenarioSet] = [("all_seed", seed_cols)]
        for left, right in combinations(seed_cols, 2):
            out.append((f"seed_pair_{left}_{right}", [left, right]))
        return out

    quantile_cols = [
        col
        for col in df.columns
        if col.startswith("pred_q") and col != "pred_q90_q10_width"
    ]
    if len(quantile_cols) >= 2:
        return [("all_quantile", sorted(quantile_cols, key=_quantile_sort_key))]

    return [("single_price", [infer_price_column(df)])]


def parse_scenario_sets(text: str, df: pd.DataFrame) -> list[ScenarioSet]:
    if not text:
        return default_scenario_sets(df)

    scenario_sets: list[ScenarioSet] = []
    for raw_spec in text.split(";"):
        spec = raw_spec.strip()
        if not spec:
            continue
        if ":" in spec:
            name, columns_text = spec.split(":", 1)
            columns = [col.strip() for col in columns_text.split(",") if col.strip()]
        else:
            columns = [col.strip() for col in spec.split(",") if col.strip()]
            name = "_".join(columns)
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"scenario set {name} has missing columns: {missing}")
        if not columns:
            raise ValueError(f"scenario set {name} is empty")
        scenario_sets.append((name.strip(), columns))
    if not scenario_sets:
        raise ValueError("no scenario sets parsed")
    return scenario_sets


def build_candidate_pool(
    price_df: pd.DataFrame,
    reference_submission: pd.DataFrame,
    scenario_sets: Sequence[ScenarioSet],
    risk_lambdas: Sequence[float],
    max_abs_start_deltas: Sequence[int],
    min_delta_score: float = 0.0,
    blocked_dates: Optional[set[str]] = None,
    price_col: str = "",
    threshold: float = 0.0,
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
) -> pd.DataFrame:
    blocked_dates = blocked_dates or set()
    inferred_price_col = price_col or infer_price_column(price_df)
    rows: list[pd.DataFrame] = []

    for scenario_name, scenario_cols in scenario_sets:
        for risk_lambda in risk_lambdas:
            _, meta = generate_stochastic_strategy(
                price_df,
                reference_submission,
                scenario_cols=list(scenario_cols),
                price_col=inferred_price_col,
                risk_lambda=float(risk_lambda),
                threshold=float(threshold),
                charge_start_min=charge_start_min,
                charge_start_max=charge_start_max,
                discharge_start_min=discharge_start_min,
                discharge_start_max=discharge_start_max,
            )
            for max_abs_start_delta in max_abs_start_deltas:
                eligible = meta.loc[
                    (~meta["same_action"])
                    & (meta["traded"])
                    & (meta["delta_score"] >= float(min_delta_score))
                    & (~meta["date"].isin(blocked_dates))
                    & (meta["max_abs_start_delta"] <= int(max_abs_start_delta))
                ].copy()
                if eligible.empty:
                    continue
                eligible["scenario_set"] = scenario_name
                eligible["scenario_cols"] = ",".join(scenario_cols)
                eligible["risk_lambda"] = float(risk_lambda)
                eligible["max_abs_start_delta_limit"] = int(max_abs_start_delta)
                rows.append(eligible)

    if not rows:
        return pd.DataFrame()

    pool = pd.concat(rows, ignore_index=True)
    pool = pool.sort_values(
        ["delta_score", "top1_top2_margin", "expected_delta_profit"],
        ascending=False,
    )
    pool = pool.drop_duplicates(
        subset=["date", "candidate_charge_start", "candidate_discharge_start"],
        keep="first",
    ).reset_index(drop=True)
    pool.insert(0, "pool_rank", np.arange(1, len(pool) + 1))
    return pool


def add_seed_delta_diagnostics(
    pool: pd.DataFrame,
    price_df: pd.DataFrame,
    seed_cols: Optional[Sequence[str]] = None,
    block_size: int = 8,
    power_value: float = 1000.0,
) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()
    cols = list(seed_cols or [col for col in price_df.columns if col.startswith("pred_price_seed")])
    if not cols:
        return pool.copy()

    df = price_df.copy()
    if "times" not in df.columns:
        raise ValueError("price data missing required column: times")
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"seed diagnostic columns not found: {missing}")
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)
    by_date = {date: group.sort_values("times").reset_index(drop=True) for date, group in df.groupby("date")}

    out = pool.copy()
    seed_delta_rows: list[dict[str, float]] = []
    for _, row in out.iterrows():
        date = str(row["date"])
        if date not in by_date:
            raise ValueError(f"price data missing date for seed diagnostics: {date}")
        day = by_date[date]
        if len(day) != 96:
            raise ValueError(f"{date} must contain 96 rows for seed diagnostics, got {len(day)}")
        baseline_charge = int(row["baseline_charge_start"])
        baseline_discharge = int(row["baseline_discharge_start"])
        candidate_charge = int(row["candidate_charge_start"])
        candidate_discharge = int(row["candidate_discharge_start"])
        deltas: list[float] = []
        values: dict[str, float] = {}
        for col in cols:
            baseline_profit = power_value * (
                day.loc[baseline_discharge : baseline_discharge + block_size - 1, col].sum()
                - day.loc[baseline_charge : baseline_charge + block_size - 1, col].sum()
            )
            candidate_profit = power_value * (
                day.loc[candidate_discharge : candidate_discharge + block_size - 1, col].sum()
                - day.loc[candidate_charge : candidate_charge + block_size - 1, col].sum()
            )
            delta = float(candidate_profit - baseline_profit)
            values[f"{col}_delta"] = delta
            deltas.append(delta)
        values["all_seed_delta_min"] = float(np.min(deltas))
        values["all_seed_delta_mean"] = float(np.mean(deltas))
        values["all_seed_delta_positive_count"] = int(sum(delta > 0 for delta in deltas))
        seed_delta_rows.append(values)

    diagnostics = pd.DataFrame(seed_delta_rows)
    return pd.concat([out.reset_index(drop=True), diagnostics.reset_index(drop=True)], axis=1)


def filter_candidate_pool(
    pool: pd.DataFrame,
    require_scenario_set: str = "",
    min_risk_lambda: float = 0.0,
    min_top1_top2_margin: float = 0.0,
    min_all_seed_delta: Optional[float] = None,
    min_all_seed_positive_count: int = 0,
    min_submission_price_delta: Optional[float] = None,
    require_multi_price_agree: bool = False,
) -> pd.DataFrame:
    filtered = pool.copy()
    if require_scenario_set:
        filtered = filtered.loc[filtered["scenario_set"].eq(require_scenario_set)].copy()
    filtered = filtered.loc[filtered["risk_lambda"] >= float(min_risk_lambda)].copy()
    filtered = filtered.loc[filtered["top1_top2_margin"] >= float(min_top1_top2_margin)].copy()
    if min_all_seed_delta is not None:
        if "all_seed_delta_min" not in filtered.columns:
            raise ValueError("min_all_seed_delta requires seed delta diagnostics")
        filtered = filtered.loc[filtered["all_seed_delta_min"] >= float(min_all_seed_delta)].copy()
    if min_all_seed_positive_count:
        if "all_seed_delta_positive_count" not in filtered.columns:
            raise ValueError("min_all_seed_positive_count requires seed delta diagnostics")
        filtered = filtered.loc[
            filtered["all_seed_delta_positive_count"] >= int(min_all_seed_positive_count)
        ].copy()
    if min_submission_price_delta is not None:
        if "submission_price_delta" not in filtered.columns:
            raise ValueError("min_submission_price_delta requires submission price diagnostics")
        filtered = filtered.loc[
            filtered["submission_price_delta"].astype(float) >= float(min_submission_price_delta)
        ].copy()
    if require_multi_price_agree:
        if "multi_price_delta_agree" not in filtered.columns:
            raise ValueError("require_multi_price_agree requires submission price diagnostics")
        filtered = filtered.loc[filtered["multi_price_delta_agree"].astype(bool)].copy()
    sort_columns = ["delta_score", "top1_top2_margin", "expected_delta_profit"]
    if "submission_price_delta" in filtered.columns:
        sort_columns.insert(1, "submission_price_delta")
    if "all_seed_delta_min" in filtered.columns:
        sort_columns.insert(1, "all_seed_delta_min")
    return filtered.sort_values(sort_columns, ascending=False).reset_index(drop=True)


def apply_single_day_action(
    reference_submission: pd.DataFrame,
    selected: pd.Series,
    block_size: int = 8,
    power_value: float = 1000.0,
) -> pd.DataFrame:
    out = reference_submission.copy()
    if "times" not in out.columns or "power" not in out.columns:
        raise ValueError("reference submission must contain times and power columns")
    out["times"] = pd.to_datetime(out["times"])
    date = str(selected["date"])
    day_mask = out["times"].dt.date.astype(str).eq(date)
    day_indices = list(out.loc[day_mask].sort_values("times").index)
    if len(day_indices) != 96:
        raise ValueError(f"{date} must contain 96 rows in reference submission, got {len(day_indices)}")

    charge_start = int(selected["candidate_charge_start"])
    discharge_start = int(selected["candidate_discharge_start"])
    out.loc[day_indices, "power"] = 0.0
    out.loc[day_indices[charge_start : charge_start + block_size], "power"] = -float(power_value)
    out.loc[day_indices[discharge_start : discharge_start + block_size], "power"] = float(power_value)
    return out


def manifest_from_selected(
    selected: pd.Series,
    candidate_csv: str,
    candidate_sha256: str,
    reason: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_csv": candidate_csv,
                "candidate_sha256": candidate_sha256,
                "date": selected["date"],
                "blocked": False,
                "changed_days": 1,
                "baseline_charge_start": int(selected["baseline_charge_start"]),
                "baseline_discharge_start": int(selected["baseline_discharge_start"]),
                "candidate_charge_start": int(selected["candidate_charge_start"]),
                "candidate_discharge_start": int(selected["candidate_discharge_start"]),
                "pred_window_score": float(selected["pred_window_score"]),
                "baseline_window_score": float(selected["baseline_window_score"]),
                "pred_delta_score": float(selected["delta_score"]),
                "expected_delta_profit": float(selected["expected_delta_profit"]),
                "score_std": float(selected["score_std"]),
                "top1_top2_margin": float(selected["top1_top2_margin"]),
                "scenario_set": selected["scenario_set"],
                "scenario_cols": selected["scenario_cols"],
                "risk_lambda": float(selected["risk_lambda"]),
                "max_abs_start_delta_limit": int(selected["max_abs_start_delta_limit"]),
                "pool_rank": int(selected["pool_rank"]),
                "reason": reason,
                **_optional_float_fields(
                    selected,
                    [
                        "all_seed_delta_min",
                        "all_seed_delta_mean",
                        "all_seed_delta_positive_count",
                        "submission_price_delta",
                        "multi_price_delta_min",
                    ],
                ),
                **_optional_bool_fields(selected, ["multi_price_delta_agree"]),
            }
        ]
    )


def _optional_float_fields(selected: pd.Series, columns: Sequence[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for column in columns:
        if column in selected.index and not pd.isna(selected[column]):
            values[column] = float(selected[column])
    return values


def _optional_bool_fields(selected: pd.Series, columns: Sequence[str]) -> dict[str, bool]:
    values: dict[str, bool] = {}
    for column in columns:
        if column in selected.index and not pd.isna(selected[column]):
            values[column] = bool(selected[column])
    return values


def _quantile_sort_key(column: str) -> int:
    digits = "".join(ch for ch in column if ch.isdigit())
    return int(digits) if digits else 0


def _sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ranked pool of stochastic single-day candidates.")
    parser.add_argument("--price-csv", required=True)
    parser.add_argument("--reference-submission", required=True)
    parser.add_argument("--output", default="outputs/output_stochastic_pool_top1.csv")
    parser.add_argument("--pool-output", default="outputs/stochastic_candidate_pool.csv")
    parser.add_argument("--manifest-output", default="outputs/stochastic_candidate_pool_top1_manifest.csv")
    parser.add_argument("--scenario-sets", default="")
    parser.add_argument("--price-col", default="")
    parser.add_argument("--risk-lambdas", default="0,0.1,0.25,0.5")
    parser.add_argument("--max-abs-start-deltas", default="1,2,4")
    parser.add_argument("--min-delta-score", type=float, default=0.0)
    parser.add_argument("--blocked-dates", default="2026-01-11")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--require-scenario-set", default="")
    parser.add_argument("--min-risk-lambda", type=float, default=0.0)
    parser.add_argument("--min-top1-top2-margin", type=float, default=0.0)
    parser.add_argument("--min-all-seed-delta", type=float, default=None)
    parser.add_argument("--min-all-seed-positive-count", type=int, default=0)
    parser.add_argument("--submission-price-col", default="")
    parser.add_argument("--min-submission-price-delta", type=float, default=None)
    parser.add_argument("--require-multi-price-agree", action="store_true")
    parser.add_argument("--reference-score", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--reason", default="top-k stochastic candidate pool single-day replacement")
    args = parser.parse_args()

    price_df = pd.read_csv(args.price_csv)
    reference = pd.read_csv(args.reference_submission)
    scenario_sets = parse_scenario_sets(args.scenario_sets, price_df)
    pool = build_candidate_pool(
        price_df,
        reference,
        scenario_sets=scenario_sets,
        risk_lambdas=parse_float_grid(args.risk_lambdas),
        max_abs_start_deltas=parse_int_grid(args.max_abs_start_deltas),
        min_delta_score=args.min_delta_score,
        blocked_dates=parse_blocked_dates(args.blocked_dates),
        price_col=args.price_col,
        threshold=args.threshold,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
    )
    if pool.empty:
        raise SystemExit("no eligible stochastic candidates found")
    pool = add_seed_delta_diagnostics(pool, price_df)
    if args.submission_price_col:
        pool = add_submission_price_features(pool, reference, price_col=args.submission_price_col)
        predicted_delta = (
            pool["all_seed_delta_min"]
            if "all_seed_delta_min" in pool.columns
            else pool["expected_delta_profit"]
        ).astype(float)
        pool["multi_price_delta_min"] = np.minimum(
            pool["submission_price_delta"].astype(float),
            predicted_delta,
        )
        pool["multi_price_delta_agree"] = (
            (pool["submission_price_delta"].astype(float) > 0.0)
            & (predicted_delta > 0.0)
        )

    pool = filter_candidate_pool(
        pool,
        require_scenario_set=args.require_scenario_set,
        min_risk_lambda=args.min_risk_lambda,
        min_top1_top2_margin=args.min_top1_top2_margin,
        min_all_seed_delta=args.min_all_seed_delta,
        min_all_seed_positive_count=args.min_all_seed_positive_count,
        min_submission_price_delta=args.min_submission_price_delta,
        require_multi_price_agree=args.require_multi_price_agree,
    )
    if pool.empty:
        raise SystemExit("no stochastic candidates remain after conservative filters")

    pool = pool.head(int(args.top_k)).copy()
    Path(args.pool_output).parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(args.pool_output, index=False)

    selected = pool.iloc[0]
    submission = apply_single_day_action(reference, selected)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    candidate_sha = _sha256(output_path)

    manifest = manifest_from_selected(selected, output_path.as_posix(), candidate_sha, args.reason)
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    print(
        "selected_pool_candidate="
        f"rank={int(selected['pool_rank'])}, date={selected['date']}, "
        f"baseline={int(selected['baseline_charge_start'])}/{int(selected['baseline_discharge_start'])}, "
        f"candidate={int(selected['candidate_charge_start'])}/{int(selected['candidate_discharge_start'])}, "
        f"delta_score={float(selected['delta_score']):.6f}, "
        f"scenario_set={selected['scenario_set']}, risk_lambda={float(selected['risk_lambda'])}"
    )
    print(
        f"saved_submission={args.output}, pool={args.pool_output}, "
        f"manifest={args.manifest_output}, reference_score={args.reference_score}, sha256={candidate_sha}"
    )


if __name__ == "__main__":
    main()
