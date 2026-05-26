from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .storage_optimizer import infer_price_column


@dataclass
class StochasticStrategyResult:
    power: np.ndarray
    score: float
    expected_profit: float
    profit_std: float
    min_profit: float
    q10_profit: float
    top1_top2_margin: float
    charge_start: Optional[int]
    discharge_start: Optional[int]
    traded: bool


def detect_scenario_columns(df: pd.DataFrame, explicit: str = "") -> list[str]:
    if explicit:
        columns = [item.strip() for item in explicit.split(",") if item.strip()]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"scenario columns not found: {missing}")
        return columns

    seed_cols = [col for col in df.columns if col.startswith("pred_price_seed")]
    if seed_cols:
        return seed_cols

    quantile_cols = [
        col
        for col in df.columns
        if col.startswith("pred_q") and col != "pred_q90_q10_width"
    ]
    if quantile_cols:
        return sorted(quantile_cols, key=_quantile_sort_key)

    return [infer_price_column(df)]


def _quantile_sort_key(column: str) -> int:
    digits = "".join(ch for ch in column if ch.isdigit())
    return int(digits) if digits else 0


def _runsum_matrix(scenarios: np.ndarray, block_size: int) -> np.ndarray:
    return np.asarray(
        [
            scenarios[:, start : start + block_size].sum(axis=1)
            for start in range(scenarios.shape[1] - block_size + 1)
        ]
    ).T


def _prepare_scenarios(scenarios: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(scenarios, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"scenarios must be 1D or 2D, got shape={arr.shape}")
    if arr.shape[1] != 96 and arr.shape[0] == 96:
        arr = arr.T
    if arr.shape[1] == 0:
        raise ValueError("scenarios must contain at least one time step")
    return arr


def score_action_scenarios(
    scenarios: Iterable[Iterable[float]],
    charge_start: Optional[int],
    discharge_start: Optional[int],
    risk_lambda: float = 0.0,
    block_size: int = 8,
    power_value: float = 1000.0,
) -> dict[str, float]:
    arr = _prepare_scenarios(scenarios)
    if charge_start is None or discharge_start is None:
        profits = np.zeros(arr.shape[0], dtype=float)
    else:
        charge_start = int(charge_start)
        discharge_start = int(discharge_start)
        charge = arr[:, charge_start : charge_start + block_size].sum(axis=1)
        discharge = arr[:, discharge_start : discharge_start + block_size].sum(axis=1)
        profits = power_value * (discharge - charge)

    expected = float(np.mean(profits))
    std = float(np.std(profits))
    return {
        "score": float(expected - risk_lambda * std),
        "expected_profit": expected,
        "profit_std": std,
        "min_profit": float(np.min(profits)),
        "q10_profit": float(np.quantile(profits, 0.10)),
    }


def optimize_one_day_scenarios(
    scenarios: Iterable[Iterable[float]],
    threshold: float = 0.0,
    risk_lambda: float = 0.0,
    block_size: int = 8,
    power_value: float = 1000.0,
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
    min_scenario_profit: Optional[float] = None,
) -> StochasticStrategyResult:
    arr = _prepare_scenarios(scenarios)
    if arr.shape[1] != 96:
        raise ValueError(f"one day must contain 96 price points, got {arr.shape[1]}")

    block_sum = _runsum_matrix(arr, block_size)
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

    best_score = -np.inf
    second_score = -np.inf
    best_stats: dict[str, float] = {}
    best_tc: Optional[int] = None
    best_td: Optional[int] = None

    for tc in range(c_min, c_max + 1):
        for td in range(max(tc + block_size, d_min), d_max + 1):
            profits = power_value * (block_sum[:, td] - block_sum[:, tc])
            expected = float(np.mean(profits))
            std = float(np.std(profits))
            score = float(expected - risk_lambda * std)
            if score > best_score:
                second_score = best_score
                best_score = score
                best_tc = tc
                best_td = td
                best_stats = {
                    "expected_profit": expected,
                    "profit_std": std,
                    "min_profit": float(np.min(profits)),
                    "q10_profit": float(np.quantile(profits, 0.10)),
                }
            elif score > second_score:
                second_score = score

    if best_tc is None or best_td is None:
        raise RuntimeError("internal optimizer error: no feasible scenario window")

    min_ok = min_scenario_profit is None or best_stats["min_profit"] >= float(min_scenario_profit)
    traded = bool(best_score > threshold and min_ok)
    power = np.zeros(96, dtype=float)
    if traded:
        power[best_tc : best_tc + block_size] = -power_value
        power[best_td : best_td + block_size] = power_value

    margin = best_score - second_score if np.isfinite(second_score) else np.inf
    return StochasticStrategyResult(
        power=power,
        score=float(best_score),
        expected_profit=float(best_stats["expected_profit"]),
        profit_std=float(best_stats["profit_std"]),
        min_profit=float(best_stats["min_profit"]),
        q10_profit=float(best_stats["q10_profit"]),
        top1_top2_margin=float(margin),
        charge_start=best_tc if traded else None,
        discharge_start=best_td if traded else None,
        traded=traded,
    )


def _daily_windows(submission: pd.DataFrame) -> dict[str, dict[str, object]]:
    df = submission.copy()
    if "times" not in df.columns or "power" not in df.columns:
        raise ValueError("submission must contain times and power columns")
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)

    out: dict[str, dict[str, object]] = {}
    for date, group in df.groupby("date", sort=True):
        power = group.sort_values("times")["power"].to_numpy(dtype=float)
        charge = np.flatnonzero(power < 0)
        discharge = np.flatnonzero(power > 0)
        out[date] = {
            "charge_start": int(charge[0]) if len(charge) else None,
            "discharge_start": int(discharge[0]) if len(discharge) else None,
            "traded": bool(len(charge) and len(discharge)),
        }
    return out


def generate_stochastic_strategy(
    price_df: pd.DataFrame,
    reference_submission: pd.DataFrame,
    scenario_cols: list[str],
    price_col: str,
    risk_lambda: float = 0.0,
    threshold: float = 0.0,
    charge_start_min: int = 0,
    charge_start_max: int = 80,
    discharge_start_min: int = 8,
    discharge_start_max: int = 88,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = price_df.copy()
    if "times" not in df.columns:
        raise ValueError("price data missing required column: times")
    df["times"] = pd.to_datetime(df["times"])
    df = df.sort_values("times").reset_index(drop=True)
    df["date"] = df["times"].dt.date.astype(str)
    reference_windows = _daily_windows(reference_submission)

    outputs = []
    rows = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        if len(group) != 96:
            raise ValueError(f"{date} must contain 96 rows, got {len(group)}")
        scenarios = group[scenario_cols].to_numpy(dtype=float).T
        result = optimize_one_day_scenarios(
            scenarios,
            threshold=threshold,
            risk_lambda=risk_lambda,
            charge_start_min=charge_start_min,
            charge_start_max=charge_start_max,
            discharge_start_min=discharge_start_min,
            discharge_start_max=discharge_start_max,
        )
        ref = reference_windows.get(date)
        if ref is None:
            raise ValueError(f"reference submission missing date: {date}")
        baseline_stats = score_action_scenarios(
            scenarios,
            ref["charge_start"],
            ref["discharge_start"],
            risk_lambda=risk_lambda,
        )
        same_action = (
            ref["charge_start"] == result.charge_start
            and ref["discharge_start"] == result.discharge_start
        )
        charge_delta = _delta(result.charge_start, ref["charge_start"])
        discharge_delta = _delta(result.discharge_start, ref["discharge_start"])
        rows.append(
            {
                "date": date,
                "baseline_charge_start": ref["charge_start"],
                "baseline_discharge_start": ref["discharge_start"],
                "candidate_charge_start": result.charge_start,
                "candidate_discharge_start": result.discharge_start,
                "same_action": same_action,
                "delta_charge_start": charge_delta,
                "delta_discharge_start": discharge_delta,
                "max_abs_start_delta": _max_abs_delta(charge_delta, discharge_delta),
                "pred_window_score": result.score,
                "baseline_window_score": baseline_stats["score"],
                "delta_score": result.score - baseline_stats["score"],
                "expected_profit": result.expected_profit,
                "baseline_expected_profit": baseline_stats["expected_profit"],
                "expected_delta_profit": result.expected_profit - baseline_stats["expected_profit"],
                "score_std": result.profit_std,
                "baseline_score_std": baseline_stats["profit_std"],
                "min_profit": result.min_profit,
                "q10_profit": result.q10_profit,
                "top1_top2_margin": result.top1_top2_margin,
                "traded": result.traded,
            }
        )
        outputs.append(
            pd.DataFrame(
                {
                    "times": group["times"].to_numpy(),
                    price_col: group[price_col].to_numpy(dtype=float),
                    "power": result.power,
                }
            )
        )

    return pd.concat(outputs, ignore_index=True), pd.DataFrame(rows)


def select_single_day_candidate(
    reference_submission: pd.DataFrame,
    stochastic_submission: pd.DataFrame,
    meta: pd.DataFrame,
    min_delta_score: float = 0.0,
    max_abs_start_delta: Optional[int] = 4,
    blocked_dates: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    blocked_dates = blocked_dates or set()
    eligible = meta.loc[
        (~meta["same_action"])
        & (meta["traded"])
        & (meta["delta_score"] >= float(min_delta_score))
        & (~meta["date"].isin(blocked_dates))
    ].copy()
    if max_abs_start_delta is not None:
        eligible = eligible.loc[eligible["max_abs_start_delta"] <= int(max_abs_start_delta)]
    if eligible.empty:
        raise ValueError("no eligible stochastic single-day replacement found")

    selected = eligible.sort_values(
        ["delta_score", "top1_top2_margin", "expected_delta_profit"],
        ascending=False,
    ).iloc[0]
    date = str(selected["date"])

    out = reference_submission.copy()
    out["times"] = pd.to_datetime(out["times"])
    stochastic = stochastic_submission.copy()
    stochastic["times"] = pd.to_datetime(stochastic["times"])
    out_mask = out["times"].dt.date.astype(str).eq(date)
    stoch_mask = stochastic["times"].dt.date.astype(str).eq(date)
    out.loc[out_mask, "power"] = stochastic.loc[stoch_mask, "power"].to_numpy(dtype=float)
    return out, selected


def _delta(candidate: object, baseline: object) -> Optional[int]:
    if candidate is None or baseline is None or pd.isna(candidate) or pd.isna(baseline):
        return None
    return int(candidate) - int(baseline)


def _max_abs_delta(*values: Optional[int]) -> Optional[int]:
    present = [abs(int(value)) for value in values if value is not None]
    return max(present) if present else None


def _sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def _manifest_row(
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
                "reason": reason,
            }
        ]
    )


def _parse_blocked_dates(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a stochastic scenario-based storage submission.")
    parser.add_argument("--price-csv", required=True)
    parser.add_argument("--reference-submission", default="outputs/output_nwp_unconstrained_online5117.csv")
    parser.add_argument("--output", default="outputs/output_stochastic_single_day.csv")
    parser.add_argument("--meta-output", default="outputs/stochastic_strategy_meta.csv")
    parser.add_argument("--manifest-output", default="outputs/stochastic_single_day_manifest.csv")
    parser.add_argument("--scenario-cols", default="")
    parser.add_argument("--price-col", default="")
    parser.add_argument("--mode", choices=["single-day", "full"], default="single-day")
    parser.add_argument("--risk-lambda", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--min-delta-score", type=float, default=0.0)
    parser.add_argument("--max-abs-start-delta", type=int, default=4)
    parser.add_argument("--blocked-dates", default="2026-01-11")
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--reason", default="stochastic scenario optimizer single-day replacement")
    args = parser.parse_args()

    price_df = pd.read_csv(args.price_csv)
    reference = pd.read_csv(args.reference_submission)
    scenario_cols = detect_scenario_columns(price_df, args.scenario_cols)
    price_col = args.price_col or infer_price_column(price_df)
    stochastic_submission, meta = generate_stochastic_strategy(
        price_df,
        reference,
        scenario_cols=scenario_cols,
        price_col=price_col,
        risk_lambda=args.risk_lambda,
        threshold=args.threshold,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta_output).parent.mkdir(parents=True, exist_ok=True)

    selected: Optional[pd.Series] = None
    if args.mode == "single-day":
        max_delta = args.max_abs_start_delta if args.max_abs_start_delta >= 0 else None
        final_submission, selected = select_single_day_candidate(
            reference,
            stochastic_submission,
            meta,
            min_delta_score=args.min_delta_score,
            max_abs_start_delta=max_delta,
            blocked_dates=_parse_blocked_dates(args.blocked_dates),
        )
        final_submission.to_csv(output_path, index=False)
    else:
        stochastic_submission.to_csv(output_path, index=False)

    meta.to_csv(args.meta_output, index=False)
    candidate_sha = _sha256(output_path)

    if args.mode == "single-day" and selected is not None:
        manifest = _manifest_row(selected, output_path.as_posix(), candidate_sha, args.reason)
        manifest_path = Path(args.manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        print(
            "selected_single_day="
            f"{selected['date']}, baseline={int(selected['baseline_charge_start'])}/"
            f"{int(selected['baseline_discharge_start'])}, candidate="
            f"{int(selected['candidate_charge_start'])}/{int(selected['candidate_discharge_start'])}, "
            f"delta_score={float(selected['delta_score']):.6f}"
        )

    print(
        f"saved_submission={args.output}, rows={len(pd.read_csv(output_path))}, "
        f"mode={args.mode}, scenario_cols={','.join(scenario_cols)}, sha256={candidate_sha}"
    )


if __name__ == "__main__":
    main()
