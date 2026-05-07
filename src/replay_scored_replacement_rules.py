from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .replacement_classifier import (
    DISCHARGE_PLATEAU_STRENGTH_COL,
    DISCHARGE_SPIKE_RISK_COL,
    PRED_EXPECTED_DELTA_COL,
    PRED_PROBA_COL,
    RISK_EXPECTED_DELTA_COL,
    RISK_PROBA_COL,
    aggregate_replacement_metrics,
    select_daily_replacements,
)
from .rolling_validate_replacement_classifier import aggregate_by_fold


DEFAULT_FOLD_RANGES = {
    1: ("2025-04-01", "2025-04-30"),
    2: ("2025-07-01", "2025-07-31"),
    3: ("2025-10-01", "2025-10-31"),
    4: ("2025-12-01", "2025-12-31"),
}


def _load_fold_frames(scored_dir: str | Path) -> list[tuple[int, pd.DataFrame]]:
    root = Path(scored_dir)
    frames: list[tuple[int, pd.DataFrame]] = []
    for path in sorted(root.glob("fold_*_scored_windows.csv")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            fold = int(parts[1])
        except ValueError:
            continue
        frame = pd.read_csv(path)
        frame["date"] = frame["date"].astype(str)
        frames.append((fold, frame))
    if not frames:
        raise ValueError(f"no fold_*_scored_windows.csv files found in {root}")
    return frames


def _stage1_eligible(
    group: pd.DataFrame,
    proba_threshold: float,
    min_expected_delta: float,
    min_margin: float,
    require_baseline_stability: bool,
) -> pd.DataFrame:
    eligible = group.loc[
        (group[PRED_PROBA_COL].astype(float) >= float(proba_threshold))
        & (group[PRED_EXPECTED_DELTA_COL].astype(float) >= float(min_expected_delta))
        & (group["top1_minus_top2_margin"].astype(float) >= float(min_margin))
    ].copy()
    if require_baseline_stability:
        if "baseline_stability_pass" not in eligible.columns:
            raise ValueError("baseline stability requested but column baseline_stability_pass is missing")
        eligible = eligible.loc[eligible["baseline_stability_pass"].astype(float) >= 1.0].copy()
    return eligible


def _apply_shape_post_filter(
    frame: pd.DataFrame,
    spike_max: float | None,
    plateau_min: float | None,
    balance_min: float | None,
) -> pd.DataFrame:
    if spike_max is None and plateau_min is None and balance_min is None:
        return frame
    required = {DISCHARGE_SPIKE_RISK_COL, DISCHARGE_PLATEAU_STRENGTH_COL}
    if balance_min is not None:
        required.add("discharge_shape_risk_balance")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"shape post-filter requested but columns are missing: {sorted(missing)}")

    out = frame.copy()
    passed = pd.Series(True, index=out.index)
    if spike_max is not None:
        passed &= out[DISCHARGE_SPIKE_RISK_COL].astype(float) <= float(spike_max)
    if plateau_min is not None:
        passed &= out[DISCHARGE_PLATEAU_STRENGTH_COL].astype(float) >= float(plateau_min)
    if balance_min is not None:
        passed &= out["discharge_shape_risk_balance"].astype(float) >= float(balance_min)

    out["post_shape_gate_pass"] = passed.astype(float)
    out["post_shape_spike_margin"] = (
        float(spike_max) - out[DISCHARGE_SPIKE_RISK_COL].astype(float)
        if spike_max is not None
        else np.nan
    )
    out["post_shape_plateau_margin"] = (
        out[DISCHARGE_PLATEAU_STRENGTH_COL].astype(float) - float(plateau_min)
        if plateau_min is not None
        else np.nan
    )
    out["post_shape_balance_margin"] = (
        out["discharge_shape_risk_balance"].astype(float) - float(balance_min)
        if balance_min is not None
        else np.nan
    )
    out.loc[~passed, RISK_PROBA_COL] = 0.0
    out.loc[~passed, RISK_EXPECTED_DELTA_COL] = -999.0
    return out


def _diagnostic_stage1_rows(
    frames: list[tuple[int, pd.DataFrame]],
    proba_threshold: float,
    min_expected_delta: float,
    min_margin: float,
    require_baseline_stability: bool,
    source_name: str,
    spike_max: float | None,
    plateau_min: float | None,
    balance_min: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, frame in frames:
        frame = _apply_shape_post_filter(
            frame,
            spike_max=spike_max,
            plateau_min=plateau_min,
            balance_min=balance_min,
        )
        for date, group in frame.groupby("date", sort=True):
            eligible = _stage1_eligible(
                group,
                proba_threshold=proba_threshold,
                min_expected_delta=min_expected_delta,
                min_margin=min_margin,
                require_baseline_stability=require_baseline_stability,
            )
            if (
                (spike_max is not None or plateau_min is not None or balance_min is not None)
                and "post_shape_gate_pass" in eligible.columns
            ):
                eligible = eligible.loc[eligible["post_shape_gate_pass"].astype(float) >= 1.0].copy()
            if eligible.empty:
                continue
            selected = eligible.sort_values(
                [PRED_EXPECTED_DELTA_COL, PRED_PROBA_COL, "top1_minus_top2_margin"],
                ascending=[False, False, False],
            ).iloc[0]
            rows.append(
                {
                    "source": source_name,
                    "fold": int(fold),
                    "date": str(date),
                    "baseline_charge_start": int(selected["baseline_charge_start"]),
                    "baseline_discharge_start": int(selected["baseline_discharge_start"]),
                    "candidate_charge_start": int(selected["charge_start"]),
                    "candidate_discharge_start": int(selected["discharge_start"]),
                    "true_delta_profit": float(selected.get("true_delta_profit", np.nan)),
                    PRED_PROBA_COL: float(selected.get(PRED_PROBA_COL, 0.0)),
                    PRED_EXPECTED_DELTA_COL: float(selected.get(PRED_EXPECTED_DELTA_COL, 0.0)),
                    "top1_minus_top2_margin": float(selected.get("top1_minus_top2_margin", 0.0)),
                    RISK_PROBA_COL: float(selected.get(RISK_PROBA_COL, 0.0)),
                    RISK_EXPECTED_DELTA_COL: float(selected.get(RISK_EXPECTED_DELTA_COL, 0.0)),
                    "risk_rule_net_load_margin": float(selected.get("risk_rule_net_load_margin", np.nan)),
                    "risk_rule_hist_centered_margin": float(
                        selected.get("risk_rule_hist_centered_margin", np.nan)
                    ),
                    "risk_rule_discharge_spike_margin": float(
                        selected.get("risk_rule_discharge_spike_margin", np.nan)
                    ),
                    "risk_rule_discharge_plateau_margin": float(
                        selected.get("risk_rule_discharge_plateau_margin", np.nan)
                    ),
                    "risk_rule_discharge_shape_balance_margin": float(
                        selected.get("risk_rule_discharge_shape_balance_margin", np.nan)
                    ),
                    DISCHARGE_SPIKE_RISK_COL: float(selected.get(DISCHARGE_SPIKE_RISK_COL, np.nan)),
                    DISCHARGE_PLATEAU_STRENGTH_COL: float(
                        selected.get(DISCHARGE_PLATEAU_STRENGTH_COL, np.nan)
                    ),
                    "discharge_shape_risk_balance": float(
                        selected.get("discharge_shape_risk_balance", np.nan)
                    ),
                    "post_shape_gate_pass": float(selected.get("post_shape_gate_pass", np.nan)),
                    "post_shape_spike_margin": float(selected.get("post_shape_spike_margin", np.nan)),
                    "post_shape_plateau_margin": float(
                        selected.get("post_shape_plateau_margin", np.nan)
                    ),
                    "post_shape_balance_margin": float(
                        selected.get("post_shape_balance_margin", np.nan)
                    ),
                }
            )
    return pd.DataFrame(rows)


def replay_scored_dir(
    scored_dir: str | Path,
    source_name: str,
    proba_threshold: float,
    min_expected_delta: float,
    min_margin: float,
    risk_proba_threshold: float,
    min_risk_expected_delta: float,
    require_baseline_stability: bool,
    spike_max: float | None,
    plateau_min: float | None,
    balance_min: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = _load_fold_frames(scored_dir)
    details: list[pd.DataFrame] = []
    for fold, frame in frames:
        frame = _apply_shape_post_filter(
            frame,
            spike_max=spike_max,
            plateau_min=plateau_min,
            balance_min=balance_min,
        )
        day_metrics = select_daily_replacements(
            frame,
            proba_threshold=proba_threshold,
            min_expected_delta=min_expected_delta,
            min_margin=min_margin,
            risk_proba_threshold=risk_proba_threshold,
            min_risk_expected_delta=min_risk_expected_delta,
            require_baseline_stability=require_baseline_stability,
        )
        start, end = DEFAULT_FOLD_RANGES.get(fold, ("", ""))
        day_metrics.insert(0, "fold", int(fold))
        day_metrics.insert(1, "val_start", start)
        day_metrics.insert(2, "val_end", end)
        day_metrics.insert(3, "source", source_name)
        details.append(day_metrics)
    detail = pd.concat(details, ignore_index=True)
    aggregate = aggregate_by_fold(detail.drop(columns=["source"]))
    aggregate.insert(0, "source", source_name)
    stage1 = _diagnostic_stage1_rows(
        frames,
        proba_threshold=proba_threshold,
        min_expected_delta=min_expected_delta,
        min_margin=min_margin,
        require_baseline_stability=require_baseline_stability,
        source_name=source_name,
        spike_max=spike_max,
        plateau_min=plateau_min,
        balance_min=balance_min,
    )
    return detail, aggregate, stage1


def _stage1_stability(stage1: pd.DataFrame) -> pd.DataFrame:
    if stage1.empty:
        return pd.DataFrame()
    grouped = []
    for (fold, date), group in stage1.groupby(["fold", "date"], sort=True):
        candidates = {
            f"{int(row.candidate_charge_start)}/{int(row.candidate_discharge_start)}"
            for row in group.itertuples(index=False)
        }
        baselines = {
            f"{int(row.baseline_charge_start)}/{int(row.baseline_discharge_start)}"
            for row in group.itertuples(index=False)
        }
        grouped.append(
            {
                "fold": int(fold),
                "date": str(date),
                "source_count": int(group["source"].nunique()),
                "sources": ",".join(sorted(group["source"].astype(str).unique())),
                "unique_candidate_count": len(candidates),
                "candidates": ",".join(sorted(candidates)),
                "unique_baseline_count": len(baselines),
                "baselines": ",".join(sorted(baselines)),
                "all_sources_same_candidate": int(len(candidates) == 1),
                "max_true_delta_profit": float(group["true_delta_profit"].astype(float).max()),
                "min_true_delta_profit": float(group["true_delta_profit"].astype(float).min()),
            }
        )
    return pd.DataFrame(grouped)


def _stability_gate_summary(stability: pd.DataFrame, min_source_count: int) -> pd.DataFrame:
    if stability.empty:
        return pd.DataFrame(
            [
                {
                    "min_source_count": int(min_source_count),
                    "stable_days": 0,
                    "stable_positive_days": 0,
                    "stable_false_positive_days": 0,
                    "stable_total_delta_profit": 0.0,
                    "stable_worst_delta_profit": 0.0,
                    "decision": "BLOCK",
                    "reason": "no stable stage1 days",
                }
            ]
        )
    stable = stability.loc[
        (stability["source_count"].astype(int) >= int(min_source_count))
        & (stability["all_sources_same_candidate"].astype(int) == 1)
    ].copy()
    positive = stable.loc[stable["min_true_delta_profit"].astype(float) > 0.0]
    false_positive = stable.loc[stable["min_true_delta_profit"].astype(float) <= 0.0]
    stable_days = int(len(stable))
    positive_days = int(len(positive))
    false_positive_days = int(len(false_positive))
    worst_delta = float(stable["min_true_delta_profit"].min()) if stable_days else 0.0
    total_delta = float(stable["min_true_delta_profit"].sum()) if stable_days else 0.0
    decision = "PASS" if positive_days > 0 and false_positive_days == 0 else "BLOCK"
    if positive_days <= 0:
        reason = "no stable positive stage1 days"
    elif false_positive_days > 0:
        reason = "stable stage1 set includes false positives"
    else:
        reason = "stable stage1 set has positives and no false positives"
    return pd.DataFrame(
        [
            {
                "min_source_count": int(min_source_count),
                "stable_days": stable_days,
                "stable_positive_days": positive_days,
                "stable_false_positive_days": false_positive_days,
                "stable_total_delta_profit": total_delta,
                "stable_worst_delta_profit": worst_delta,
                "decision": decision,
                "reason": reason,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay conservative rules on fixed scored windows.")
    parser.add_argument(
        "--scored-dir",
        action="append",
        required=True,
        help="Directory containing fold_*_scored_windows.csv. Can be repeated.",
    )
    parser.add_argument(
        "--source-name",
        action="append",
        default=[],
        help="Optional name for each scored-dir. Must match count if supplied.",
    )
    parser.add_argument("--proba-threshold", type=float, default=0.40)
    parser.add_argument("--min-expected-delta", type=float, default=-100000.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--risk-proba-threshold", type=float, default=1.0)
    parser.add_argument("--min-risk-expected-delta", type=float, default=0.10)
    parser.add_argument("--require-baseline-stability", action="store_true")
    parser.add_argument(
        "--post-shape-spike-max",
        type=float,
        default=None,
        help="Optional hard cap for discharge_move_spike_risk_proxy before daily selection.",
    )
    parser.add_argument(
        "--post-shape-plateau-min",
        type=float,
        default=None,
        help="Optional hard floor for discharge_move_plateau_strength_proxy before daily selection.",
    )
    parser.add_argument(
        "--post-shape-balance-min",
        type=float,
        default=None,
        help="Optional hard floor for discharge_shape_risk_balance before daily selection.",
    )
    parser.add_argument("--detail-output", default="outputs/scored_rule_replay_detail.csv")
    parser.add_argument("--summary-output", default="outputs/scored_rule_replay_summary.csv")
    parser.add_argument("--stage1-output", default="outputs/scored_rule_replay_stage1.csv")
    parser.add_argument("--stage1-stability-output", default="outputs/scored_rule_replay_stage1_stability.csv")
    parser.add_argument(
        "--stage1-stability-summary-output",
        default="outputs/scored_rule_replay_stage1_stability_summary.csv",
    )
    parser.add_argument("--min-stability-source-count", type=int, default=2)
    args = parser.parse_args()

    if args.source_name and len(args.source_name) != len(args.scored_dir):
        raise ValueError("--source-name count must match --scored-dir count")
    names = args.source_name or [Path(path).name for path in args.scored_dir]

    detail_frames = []
    summary_frames = []
    stage1_frames = []
    for scored_dir, name in zip(args.scored_dir, names):
        detail, summary, stage1 = replay_scored_dir(
            scored_dir=scored_dir,
            source_name=name,
            proba_threshold=args.proba_threshold,
            min_expected_delta=args.min_expected_delta,
            min_margin=args.min_margin,
            risk_proba_threshold=args.risk_proba_threshold,
            min_risk_expected_delta=args.min_risk_expected_delta,
            require_baseline_stability=args.require_baseline_stability,
            spike_max=args.post_shape_spike_max,
            plateau_min=args.post_shape_plateau_min,
            balance_min=args.post_shape_balance_min,
        )
        detail_frames.append(detail)
        summary_frames.append(summary)
        stage1_frames.append(stage1)

    detail_out = pd.concat(detail_frames, ignore_index=True)
    summary_out = pd.concat(summary_frames, ignore_index=True)
    stage1_out = pd.concat(stage1_frames, ignore_index=True) if stage1_frames else pd.DataFrame()
    stability_out = _stage1_stability(stage1_out)
    stability_summary = _stability_gate_summary(stability_out, args.min_stability_source_count)

    Path(args.detail_output).parent.mkdir(parents=True, exist_ok=True)
    detail_out.to_csv(args.detail_output, index=False)
    summary_out.to_csv(args.summary_output, index=False)
    stage1_out.to_csv(args.stage1_output, index=False)
    stability_out.to_csv(args.stage1_stability_output, index=False)
    stability_summary.to_csv(args.stage1_stability_summary_output, index=False)

    print(f"saved_detail={args.detail_output}")
    print(f"saved_summary={args.summary_output}")
    print(f"saved_stage1={args.stage1_output}")
    print(f"saved_stage1_stability={args.stage1_stability_output}")
    print(f"saved_stage1_stability_summary={args.stage1_stability_summary_output}")
    print(summary_out.to_string(index=False))
    print(stability_summary.to_string(index=False))


if __name__ == "__main__":
    main()
