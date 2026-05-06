from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASELINE_COLS = ["date", "baseline_charge_start", "baseline_discharge_start"]


def _read_meta(path: str | Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(BASELINE_COLS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns for {label}: {sorted(missing)}")
    out = frame.loc[:, BASELINE_COLS].copy()
    out["date"] = out["date"].astype(str)
    out[f"{label}_charge_start"] = out["baseline_charge_start"].astype(int)
    out[f"{label}_discharge_start"] = out["baseline_discharge_start"].astype(int)
    return out.drop(columns=["baseline_charge_start", "baseline_discharge_start"])


def _compare_meta(left: pd.DataFrame, right: pd.DataFrame, left_label: str, right_label: str) -> pd.DataFrame:
    merged = left.merge(right, on="date", how="inner")
    merged["charge_delta"] = (
        merged[f"{right_label}_charge_start"].astype(int) - merged[f"{left_label}_charge_start"].astype(int)
    )
    merged["discharge_delta"] = (
        merged[f"{right_label}_discharge_start"].astype(int) - merged[f"{left_label}_discharge_start"].astype(int)
    )
    merged["same_baseline"] = (merged["charge_delta"] == 0) & (merged["discharge_delta"] == 0)
    merged["abs_charge_delta"] = merged["charge_delta"].abs()
    merged["abs_discharge_delta"] = merged["discharge_delta"].abs()
    merged["max_abs_delta"] = merged[["abs_charge_delta", "abs_discharge_delta"]].max(axis=1)
    return merged


def _summarize(comparison: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if comparison.empty:
        return {
            "comparison": label,
            "overlap_days": 0,
            "changed_days": 0,
            "same_days": 0,
            "changed_rate": 0.0,
            "mean_abs_charge_delta": 0.0,
            "mean_abs_discharge_delta": 0.0,
            "max_abs_charge_delta": 0.0,
            "max_abs_discharge_delta": 0.0,
            "large_drift_days_ge_8": 0,
        }
    changed = ~comparison["same_baseline"]
    return {
        "comparison": label,
        "overlap_days": int(len(comparison)),
        "changed_days": int(changed.sum()),
        "same_days": int((~changed).sum()),
        "changed_rate": float(changed.mean()),
        "mean_abs_charge_delta": float(comparison["abs_charge_delta"].mean()),
        "mean_abs_discharge_delta": float(comparison["abs_discharge_delta"].mean()),
        "max_abs_charge_delta": float(comparison["abs_charge_delta"].max()),
        "max_abs_discharge_delta": float(comparison["abs_discharge_delta"].max()),
        "large_drift_days_ge_8": int((comparison["max_abs_delta"] >= 8).sum()),
    }


def _read_optional_day_metrics(path: str | Path, source_label: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    day = pd.read_csv(path)
    if "date" not in day.columns:
        raise ValueError(f"{path} missing date column")
    day = day.copy()
    day["date"] = day["date"].astype(str)
    keep = [
        "date",
        "proposed",
        "selected_true_delta_profit",
        "baseline_charge_start",
        "baseline_discharge_start",
        "candidate_charge_start",
        "candidate_discharge_start",
        "risk_expected_delta",
        "pred_positive_proba",
    ]
    keep = [col for col in keep if col in day.columns]
    out = day.loc[:, keep].copy()
    return out.add_prefix(f"{source_label}_").rename(columns={f"{source_label}_date": "date"})


def _summarize_test_windows(path: str | Path) -> dict[str, float | int | str]:
    if not path:
        return {}
    frame = pd.read_csv(path)
    has_date = "date" in frame.columns
    pred_rank_1 = (
        frame["pred_rank"].astype(float).eq(1.0)
        if "pred_rank" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    summary: dict[str, float | int | str] = {
        "comparison": "test_candidate_gate",
        "rows": int(len(frame)),
        "days": int(frame["date"].nunique()) if has_date else 0,
        "pred_rank_1_rows": int(pred_rank_1.sum()),
        "pred_rank_1_days": int(frame.loc[pred_rank_1, "date"].nunique()) if has_date else 0,
    }
    if "risk_rule_structural_pass" in frame.columns:
        structural = frame["risk_rule_structural_pass"].astype(float).eq(1.0)
        summary["structural_pass_rows"] = int(structural.sum())
        summary["structural_pass_days"] = int(frame.loc[structural, "date"].nunique()) if has_date else 0
    if "risk_expected_delta" in frame.columns:
        risk = frame["risk_expected_delta"].astype(float)
        summary["risk_ge_0_rows"] = int((risk >= 0.0).sum())
        summary["risk_ge_0_days"] = int(frame.loc[risk >= 0.0, "date"].nunique()) if has_date else 0
        summary["risk_ge_010_rows"] = int((risk >= 0.10).sum())
        summary["risk_ge_010_days"] = int(frame.loc[risk >= 0.10, "date"].nunique()) if has_date else 0
        summary["max_risk_expected_delta"] = float(risk.max())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare safe5117-source baseline windows across rolling, full-train, and test gates."
    )
    parser.add_argument("--rolling-meta", required=True)
    parser.add_argument("--full-meta", required=True)
    parser.add_argument("--rolling-label", default="rolling")
    parser.add_argument("--full-label", default="full")
    parser.add_argument("--rolling-day-metrics", default="")
    parser.add_argument("--full-day-metrics", default="")
    parser.add_argument("--test-windows", default="")
    parser.add_argument("--detail-output", default="")
    parser.add_argument("--summary-output", default="")
    args = parser.parse_args()

    rolling = _read_meta(args.rolling_meta, args.rolling_label)
    full = _read_meta(args.full_meta, args.full_label)
    comparison = _compare_meta(rolling, full, args.rolling_label, args.full_label)

    rolling_days = _read_optional_day_metrics(args.rolling_day_metrics, "rolling")
    full_days = _read_optional_day_metrics(args.full_day_metrics, "full")
    if not rolling_days.empty:
        comparison = comparison.merge(rolling_days, on="date", how="left")
    if not full_days.empty:
        comparison = comparison.merge(full_days, on="date", how="left")

    summary_rows = [_summarize(comparison, f"{args.rolling_label}_vs_{args.full_label}")]
    test_summary = _summarize_test_windows(args.test_windows)
    if test_summary:
        summary_rows.append(test_summary)
    summary = pd.DataFrame(summary_rows)

    print(summary.to_string(index=False))
    changed = comparison.loc[~comparison["same_baseline"]].copy()
    if not changed.empty:
        print(
            changed.sort_values(["max_abs_delta", "date"], ascending=[False, True])
            .head(20)
            .to_string(index=False)
        )

    if args.detail_output:
        detail_path = Path(args.detail_output)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(detail_path, index=False)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
