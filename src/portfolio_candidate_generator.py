from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .offline_policy_improvement import (
    add_submission_price_features,
    baseline_windows_from_submission_frame,
)
from .stochastic_candidate_pool import add_seed_delta_diagnostics
from .storage_optimizer import optimize_one_day


BLOCK_SIZE = 8
POWER_VALUE = 1000.0


def parse_int_grid(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("integer grid must contain at least one value")
    return values


def parse_blocked_dates(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def infer_submission_price_col(submission: pd.DataFrame) -> str:
    for column in submission.columns:
        if column not in {"times", "power"}:
            return column
    raise ValueError("submission must contain a non-times/non-power price column")


def build_calendar_oracle_windows(train_label: pd.DataFrame, target_col: str = "A") -> pd.DataFrame:
    if "times" not in train_label.columns or target_col not in train_label.columns:
        raise ValueError("train label must contain times and target column")
    df = train_label.copy()
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)
    df["month_day"] = df["times"].dt.strftime("%m-%d")

    rows: list[dict[str, object]] = []
    for month_day, group in df.groupby("month_day", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        if len(group) != 96:
            continue
        result = optimize_one_day(
            group[target_col].to_numpy(dtype=float),
            threshold=0.0,
            block_size=BLOCK_SIZE,
            power_value=POWER_VALUE,
        )
        if not result.traded:
            continue
        rows.append(
            {
                "month_day": month_day,
                "hist_date": str(group["date"].iloc[0]),
                "hist_charge_start": int(result.charge_start),
                "hist_discharge_start": int(result.discharge_start),
                "hist_oracle_profit": float(result.best_profit),
            }
        )
    if not rows:
        raise ValueError("calendar oracle window extraction produced no rows")
    return pd.DataFrame(rows)


def move_toward_baseline(
    baseline: int,
    target: int,
    cap: int | None,
    min_start: int,
    max_start: int,
) -> int:
    if cap is None:
        value = target
    else:
        delta = int(np.clip(target - baseline, -cap, cap))
        value = baseline + delta
    return int(np.clip(value, min_start, max_start))


def make_feasible_windows(actions: pd.DataFrame) -> pd.DataFrame:
    out = actions.copy()
    max_charge_start = 96 - 2 * BLOCK_SIZE
    max_start = 96 - BLOCK_SIZE
    out["candidate_charge_start"] = out["candidate_charge_start"].clip(0, max_charge_start).astype(int)
    out["candidate_discharge_start"] = (
        out["candidate_discharge_start"].clip(BLOCK_SIZE, max_start).astype(int)
    )
    too_early = out["candidate_discharge_start"] < out["candidate_charge_start"] + BLOCK_SIZE
    out.loc[too_early, "candidate_discharge_start"] = (
        out.loc[too_early, "candidate_charge_start"] + BLOCK_SIZE
    ).clip(upper=max_start)
    return out


def build_calendar_action_pool(
    reference_submission: pd.DataFrame,
    calendar_windows: pd.DataFrame,
    cap_shift: int | None,
    blocked_dates: set[str],
) -> pd.DataFrame:
    baseline = baseline_windows_from_submission_frame(reference_submission)
    baseline["month_day"] = pd.to_datetime(baseline["date"]).dt.strftime("%m-%d")
    actions = baseline.merge(calendar_windows, on="month_day", how="left")
    actions = actions.dropna(
        subset=["hist_charge_start", "hist_discharge_start"]
    ).copy()
    actions = actions.loc[~actions["date"].isin(blocked_dates)].copy()

    charge_values: list[int] = []
    discharge_values: list[int] = []
    for _, row in actions.iterrows():
        charge = move_toward_baseline(
            int(row["baseline_charge_start"]),
            int(row["hist_charge_start"]),
            cap_shift,
            0,
            96 - 2 * BLOCK_SIZE,
        )
        discharge = move_toward_baseline(
            int(row["baseline_discharge_start"]),
            int(row["hist_discharge_start"]),
            cap_shift,
            BLOCK_SIZE,
            96 - BLOCK_SIZE,
        )
        charge_values.append(charge)
        discharge_values.append(discharge)
    actions["candidate_charge_start"] = charge_values
    actions["candidate_discharge_start"] = discharge_values
    actions = make_feasible_windows(actions)
    actions["delta_charge_start"] = (
        actions["candidate_charge_start"].astype(int)
        - actions["baseline_charge_start"].astype(int)
    )
    actions["delta_discharge_start"] = (
        actions["candidate_discharge_start"].astype(int)
        - actions["baseline_discharge_start"].astype(int)
    )
    actions["max_abs_start_delta"] = actions[
        ["delta_charge_start", "delta_discharge_start"]
    ].abs().max(axis=1)
    actions["total_abs_start_delta"] = actions[
        ["delta_charge_start", "delta_discharge_start"]
    ].abs().sum(axis=1)
    actions["same_action"] = (
        (actions["delta_charge_start"] == 0)
        & (actions["delta_discharge_start"] == 0)
    )
    actions["cap_shift"] = -1 if cap_shift is None else int(cap_shift)
    return actions.loc[~actions["same_action"]].reset_index(drop=True)


def add_residual_scenario_diagnostics(
    actions: pd.DataFrame,
    residual_df: pd.DataFrame,
    scenario_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    cols = list(
        scenario_cols
        or [column for column in residual_df.columns if column.startswith("resid_scenario_")]
    )
    if not cols:
        return actions.copy()

    df = residual_df.copy()
    if "times" not in df.columns:
        raise ValueError("residual scenario data missing times column")
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)
    by_date = {
        date: group.sort_values("times").reset_index(drop=True)
        for date, group in df.groupby("date", sort=True)
    }

    rows: list[dict[str, float]] = []
    for _, row in actions.iterrows():
        date = str(row["date"])
        if date not in by_date:
            raise ValueError(f"residual scenario data missing date: {date}")
        day = by_date[date]
        if len(day) != 96:
            raise ValueError(f"{date} must contain 96 residual scenario rows")
        deltas: list[float] = []
        for column in cols:
            prices = day[column].to_numpy(dtype=float)
            baseline_profit = _window_profit(
                prices,
                int(row["baseline_charge_start"]),
                int(row["baseline_discharge_start"]),
            )
            candidate_profit = _window_profit(
                prices,
                int(row["candidate_charge_start"]),
                int(row["candidate_discharge_start"]),
            )
            deltas.append(candidate_profit - baseline_profit)
        arr = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "residual_delta_min": float(np.min(arr)),
                "residual_delta_p10": float(np.quantile(arr, 0.10)),
                "residual_delta_mean": float(np.mean(arr)),
                "residual_delta_std": float(np.std(arr)),
                "residual_positive_rate": float(np.mean(arr > 0.0)),
            }
        )
    return pd.concat([actions.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_calendar_replay_diagnostics(
    actions: pd.DataFrame,
    train_label: pd.DataFrame,
    target_col: str = "A",
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    if "times" not in train_label.columns or target_col not in train_label.columns:
        raise ValueError("train label must contain times and target column")

    labels = train_label.copy()
    labels["times"] = pd.to_datetime(labels["times"])
    labels["month_day"] = labels["times"].dt.strftime("%m-%d")
    by_month_day = {
        month_day: group.sort_values("times").reset_index(drop=True)
        for month_day, group in labels.groupby("month_day", sort=True)
    }

    rows: list[dict[str, float]] = []
    for _, row in actions.iterrows():
        month_day = str(row["month_day"])
        if month_day not in by_month_day:
            raise ValueError(f"train label missing month-day: {month_day}")
        day = by_month_day[month_day]
        if len(day) != 96:
            raise ValueError(f"{month_day} must contain 96 historical rows")
        prices = day[target_col].to_numpy(dtype=float)
        baseline_profit = _window_profit(
            prices,
            int(row["baseline_charge_start"]),
            int(row["baseline_discharge_start"]),
        )
        candidate_profit = _window_profit(
            prices,
            int(row["candidate_charge_start"]),
            int(row["candidate_discharge_start"]),
        )
        rows.append(
            {
                "hist_baseline_profit": float(baseline_profit),
                "hist_candidate_profit": float(candidate_profit),
                "hist_delta_profit": float(candidate_profit - baseline_profit),
            }
        )
    return pd.concat([actions.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_portfolio_rank_score(actions: pd.DataFrame) -> pd.DataFrame:
    out = actions.copy()
    if "all_seed_delta_min" not in out.columns:
        out["all_seed_delta_min"] = 0.0
    if "residual_delta_p10" not in out.columns:
        out["residual_delta_p10"] = 0.0
    if "submission_price_delta" not in out.columns:
        out["submission_price_delta"] = 0.0
    out["portfolio_rank_score"] = (
        0.40 * out["all_seed_delta_min"].astype(float)
        + 0.35 * out["residual_delta_p10"].astype(float)
        + 0.05 * out["submission_price_delta"].astype(float)
        - 25.0 * out["max_abs_start_delta"].astype(float)
    )
    return out.sort_values(
        [
            "portfolio_rank_score",
            "all_seed_delta_min",
            "residual_delta_p10",
            "submission_price_delta",
        ],
        ascending=False,
    ).reset_index(drop=True)


def apply_portfolio_actions(
    reference_submission: pd.DataFrame,
    selected_actions: pd.DataFrame,
) -> pd.DataFrame:
    out = reference_submission.copy()
    out["times"] = pd.to_datetime(out["times"])
    selected_by_date = {str(row["date"]): row for _, row in selected_actions.iterrows()}
    for date, row in selected_by_date.items():
        mask = out["times"].dt.date.astype(str).eq(date)
        day_indices = list(out.loc[mask].sort_values("times").index)
        if len(day_indices) != 96:
            raise ValueError(f"{date} must contain 96 rows in reference submission")
        charge_start = int(row["candidate_charge_start"])
        discharge_start = int(row["candidate_discharge_start"])
        out.loc[day_indices, "power"] = 0.0
        out.loc[day_indices[charge_start : charge_start + BLOCK_SIZE], "power"] = -POWER_VALUE
        out.loc[
            day_indices[discharge_start : discharge_start + BLOCK_SIZE],
            "power",
        ] = POWER_VALUE
    return out


def build_candidate_files(
    action_pool: pd.DataFrame,
    reference_submission: pd.DataFrame,
    top_counts: Sequence[int],
    output_dir: Path,
    prefix: str,
    reference_score: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    pool = action_pool.sort_values("portfolio_rank_score", ascending=False).reset_index(drop=True)
    reference_day_count = _reference_day_count(reference_submission)

    for top_count in sorted(set(int(value) for value in top_counts)):
        selected = pool.head(top_count).copy()
        candidate_name = f"{prefix}_top{top_count}"
        output_path = output_dir / f"output_{candidate_name}.csv"
        submission = apply_portfolio_actions(reference_submission, selected)
        submission.to_csv(output_path, index=False)
        sha = sha256(output_path)
        selected_dates = ",".join(str(value) for value in selected["date"].tolist())
        selected_actions = "; ".join(
            (
                f"{row['date']}:{int(row['baseline_charge_start'])}/"
                f"{int(row['baseline_discharge_start'])}->"
                f"{int(row['candidate_charge_start'])}/"
                f"{int(row['candidate_discharge_start'])}"
            )
            for _, row in selected.iterrows()
        )
        seed_min_sum = float(selected["all_seed_delta_min"].sum())
        seed_mean_sum = float(selected["all_seed_delta_mean"].sum())
        residual_p10_sum = float(selected.get("residual_delta_p10", pd.Series(dtype=float)).sum())
        submission_delta_sum = float(selected["submission_price_delta"].sum())
        hist_delta_sum = _optional_sum(selected, "hist_delta_profit")
        hist_replay_estimated_score = (
            float(reference_score + hist_delta_sum / reference_day_count)
            if pd.notna(hist_delta_sum)
            else np.nan
        )
        portfolio_score = (
            0.40 * seed_min_sum
            + 0.35 * residual_p10_sum
            + 0.05 * submission_delta_sum
            - 25.0 * float(selected["max_abs_start_delta"].sum())
        )
        row = {
            "candidate": candidate_name,
            "candidate_csv": output_path.as_posix(),
            "candidate_sha256": sha,
            "changed_days": int(len(selected)),
            "reference_score": float(reference_score),
            "selected_dates": selected_dates,
            "selected_actions": selected_actions,
            "seed_delta_min_sum": seed_min_sum,
            "seed_delta_mean_sum": seed_mean_sum,
            "residual_delta_p10_sum": residual_p10_sum,
            "submission_price_delta_sum": submission_delta_sum,
            "hist_2025_same_day_delta_sum": hist_delta_sum,
            "hist_replay_estimated_score": hist_replay_estimated_score,
            "portfolio_score": float(portfolio_score),
            "max_abs_start_delta_max": int(selected["max_abs_start_delta"].max()),
            "max_abs_start_delta_mean": float(selected["max_abs_start_delta"].mean()),
            "negative_submission_delta_days": int((selected["submission_price_delta"] < 0.0).sum()),
            "reason": (
                "high-upside calendar oracle portfolio; intentionally changes multiple days "
                "because single-day probes are no longer enough for the 8000 target"
            ),
        }
        summary_rows.append(row)
        manifest_rows.append(
            {
                **row,
                "manifest_stage": "portfolio_high_upside",
                "blocked": False,
                "portfolio_acknowledged_high_risk": True,
                "allow_negative_submission_delta": True,
                "multi_price_delta_agree": submission_delta_sum > 0.0 and seed_min_sum > 0.0,
                "submission_price_delta": submission_delta_sum,
                "pred_window_score": seed_min_sum,
                "expected_delta_profit": seed_mean_sum,
                "hist_2025_same_day_delta_sum": hist_delta_sum,
                "hist_replay_estimated_score": hist_replay_estimated_score,
                "score_std": float(selected["all_seed_delta_min"].std(ddof=0)),
                "top1_top2_margin": float(selected["portfolio_rank_score"].min()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["portfolio_score", "seed_delta_min_sum", "residual_delta_p10_sum"],
        ascending=False,
    )
    manifest = pd.DataFrame(manifest_rows)
    return summary.reset_index(drop=True), manifest.reset_index(drop=True)


def _reference_day_count(reference_submission: pd.DataFrame) -> int:
    if "times" not in reference_submission.columns:
        raise ValueError("reference submission must contain times")
    times = pd.to_datetime(reference_submission["times"])
    day_count = int(times.dt.date.nunique())
    if day_count <= 0:
        raise ValueError("reference submission must contain at least one day")
    return day_count


def _optional_sum(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    return float(df[column].sum())


def expand_top_counts_for_target(
    action_pool: pd.DataFrame,
    top_counts: Sequence[int],
    reference_submission: pd.DataFrame,
    reference_score: float,
    target_score: float | None,
) -> list[int]:
    counts = {int(value) for value in top_counts}
    if target_score is None or target_score <= 0.0 or "hist_delta_profit" not in action_pool.columns:
        return sorted(counts)

    pool = action_pool.sort_values("portfolio_rank_score", ascending=False).reset_index(drop=True)
    day_count = _reference_day_count(reference_submission)
    cumulative = pool["hist_delta_profit"].astype(float).cumsum()
    estimated = reference_score + cumulative / day_count
    crossing = estimated.loc[estimated >= float(target_score)]
    if crossing.empty:
        return sorted(counts)

    target_count = int(crossing.index[0]) + 1
    for value in (target_count - 1, target_count, target_count + 1):
        if value > 0:
            counts.add(value)
    return sorted(counts)


def _window_profit(prices: np.ndarray, charge_start: int, discharge_start: int) -> float:
    return float(
        POWER_VALUE
        * (
            prices[discharge_start : discharge_start + BLOCK_SIZE].sum()
            - prices[charge_start : charge_start + BLOCK_SIZE].sum()
        )
    )


def sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high-upside portfolio submission candidates.")
    parser.add_argument("--train-label", default="to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv")
    parser.add_argument("--reference-submission", default="outputs/output_stochastic_conservative_online5135_20260526.csv")
    parser.add_argument("--test-pred-csv", default="outputs/test_predictions_nwp.csv")
    parser.add_argument("--residual-scenario-csv", default="outputs/test_predictions_residual_scenarios_20260526.csv")
    parser.add_argument("--target-col", default="A")
    parser.add_argument("--blocked-dates", default="")
    parser.add_argument("--cap-shift", type=int, default=-1, help="-1 means exact historical oracle windows.")
    parser.add_argument("--top-counts", default="5,8,10,12,15")
    parser.add_argument(
        "--target-score",
        type=float,
        default=0.0,
        help="If positive, add the smallest top-N whose 2025 same-day replay estimate reaches this score.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--prefix", default="portfolio_calendar_exact_20260527")
    parser.add_argument("--action-pool-output", default="outputs/portfolio_calendar_action_pool_20260527.csv")
    parser.add_argument("--summary-output", default="outputs/portfolio_candidate_summary_20260527.csv")
    parser.add_argument("--manifest-output", default="outputs/portfolio_candidate_manifest_20260527.csv")
    parser.add_argument("--reference-score", type=float, default=5135.148567685195)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference_submission)
    train_label = pd.read_csv(args.train_label)
    calendar = build_calendar_oracle_windows(train_label, target_col=args.target_col)
    cap_shift = None if args.cap_shift < 0 else int(args.cap_shift)
    pool = build_calendar_action_pool(
        reference,
        calendar,
        cap_shift=cap_shift,
        blocked_dates=parse_blocked_dates(args.blocked_dates),
    )
    pool = add_seed_delta_diagnostics(pool, pd.read_csv(args.test_pred_csv))
    price_col = infer_submission_price_col(reference)
    pool = add_submission_price_features(pool, reference, price_col=price_col)
    residual_path = Path(args.residual_scenario_csv)
    if residual_path.exists():
        pool = add_residual_scenario_diagnostics(pool, pd.read_csv(residual_path))
    pool = add_calendar_replay_diagnostics(pool, train_label, target_col=args.target_col)
    pool = add_portfolio_rank_score(pool)
    top_counts = expand_top_counts_for_target(
        pool,
        parse_int_grid(args.top_counts),
        reference,
        reference_score=float(args.reference_score),
        target_score=float(args.target_score) if args.target_score > 0.0 else None,
    )

    action_pool_path = Path(args.action_pool_output)
    action_pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(action_pool_path, index=False)

    summary, manifest = build_candidate_files(
        pool,
        reference,
        top_counts=top_counts,
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
        reference_score=args.reference_score,
    )
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    Path(args.manifest_output).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest_output, index=False)

    if args.target_score > 0.0 and "hist_replay_estimated_score" in summary.columns:
        eligible = summary.loc[
            summary["hist_replay_estimated_score"].notna()
            & (summary["hist_replay_estimated_score"] >= float(args.target_score))
        ].sort_values(["changed_days", "submission_price_delta_sum"], ascending=[True, False])
        best = eligible.iloc[0] if not eligible.empty else summary.iloc[0]
    else:
        best = summary.iloc[0]
    print(
        "portfolio_candidate_recommendation="
        f"{best['candidate']}, csv={best['candidate_csv']}, "
        f"changed_days={int(best['changed_days'])}, "
        f"seed_delta_min_sum={float(best['seed_delta_min_sum']):.6f}, "
        f"residual_delta_p10_sum={float(best['residual_delta_p10_sum']):.6f}, "
        f"hist_replay_estimated_score={float(best.get('hist_replay_estimated_score', np.nan)):.6f}, "
        f"submission_price_delta_sum={float(best['submission_price_delta_sum']):.6f}, "
        f"sha256={best['candidate_sha256']}"
    )
    print(
        f"saved_action_pool={args.action_pool_output}, "
        f"summary={args.summary_output}, manifest={args.manifest_output}"
    )


if __name__ == "__main__":
    main()
