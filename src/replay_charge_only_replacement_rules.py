from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .replacement_classifier import (
    PRED_EXPECTED_DELTA_COL,
    PRED_PROBA_COL,
    RISK_EXPECTED_DELTA_COL,
    RISK_PROBA_COL,
)


RECENT_CENTERED_COL = "delta_vs_baseline_spread_price_hist_recent_28d_slot_mean_daily_centered"
HIST_CENTERED_COL = "delta_vs_baseline_spread_hist_slot_mean_daily_centered"


def _load_scored_folds(scored_dir: str | Path) -> list[tuple[int, pd.DataFrame]]:
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


def _base_charge_only_filter(
    frame: pd.DataFrame,
    proba_threshold: float,
    min_expected_delta: float,
    require_baseline_stability: bool,
    max_abs_charge_delta: int,
    direction: str,
    recent_centered_min: float | None,
    hist_centered_min: float | None,
    risk_proba_threshold: float | None,
    min_risk_expected_delta: float | None,
) -> pd.DataFrame:
    required = {
        "date",
        "charge_start",
        "discharge_start",
        "baseline_charge_start",
        "baseline_discharge_start",
        PRED_PROBA_COL,
        PRED_EXPECTED_DELTA_COL,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"scored windows missing required columns: {sorted(missing)}")
    if require_baseline_stability and "baseline_stability_pass" not in frame.columns:
        raise ValueError("baseline stability requested but baseline_stability_pass is missing")
    if risk_proba_threshold is not None and RISK_PROBA_COL not in frame.columns:
        raise ValueError(f"risk proba threshold requested but {RISK_PROBA_COL} is missing")
    if min_risk_expected_delta is not None and RISK_EXPECTED_DELTA_COL not in frame.columns:
        raise ValueError(f"risk expected delta threshold requested but {RISK_EXPECTED_DELTA_COL} is missing")

    out = frame.copy()
    out["charge_delta"] = out["charge_start"].astype(int) - out["baseline_charge_start"].astype(int)
    out["discharge_delta"] = (
        out["discharge_start"].astype(int) - out["baseline_discharge_start"].astype(int)
    )
    mask = (
        (out[PRED_PROBA_COL].astype(float) >= float(proba_threshold))
        & (out[PRED_EXPECTED_DELTA_COL].astype(float) >= float(min_expected_delta))
        & out["discharge_delta"].eq(0)
        & out["charge_delta"].ne(0)
        & (out["charge_delta"].abs() <= int(max_abs_charge_delta))
    )
    if direction == "earlier":
        mask &= out["charge_delta"] < 0
    elif direction == "later":
        mask &= out["charge_delta"] > 0
    elif direction != "any":
        raise ValueError(f"unknown direction: {direction}")
    if require_baseline_stability:
        mask &= out["baseline_stability_pass"].astype(float) >= 1.0
    if recent_centered_min is not None:
        if RECENT_CENTERED_COL not in out.columns:
            raise ValueError(f"recent centered threshold requested but {RECENT_CENTERED_COL} is missing")
        mask &= out[RECENT_CENTERED_COL].astype(float) >= float(recent_centered_min)
    if hist_centered_min is not None:
        if HIST_CENTERED_COL not in out.columns:
            raise ValueError(f"hist centered threshold requested but {HIST_CENTERED_COL} is missing")
        mask &= out[HIST_CENTERED_COL].astype(float) >= float(hist_centered_min)
    if risk_proba_threshold is not None:
        mask &= out[RISK_PROBA_COL].astype(float) >= float(risk_proba_threshold)
    if min_risk_expected_delta is not None:
        mask &= out[RISK_EXPECTED_DELTA_COL].astype(float) >= float(min_risk_expected_delta)
    return out.loc[mask].copy()


def _select_one_per_day(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_cols = [
        PRED_EXPECTED_DELTA_COL,
        PRED_PROBA_COL,
    ]
    if RECENT_CENTERED_COL in frame.columns:
        sort_cols.append(RECENT_CENTERED_COL)
    if HIST_CENTERED_COL in frame.columns:
        sort_cols.append(HIST_CENTERED_COL)
    return (
        frame.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        .groupby("date", as_index=False)
        .head(1)
        .sort_values("date")
        .reset_index(drop=True)
    )


def _summarize_selected(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {
            "days": 0,
            "positive_days": 0,
            "false_positive_days": 0,
            "total_delta_profit": 0.0,
            "avg_delta_profit": 0.0,
            "worst_delta_profit": 0.0,
        }
    true_delta = selected.get("true_delta_profit")
    if true_delta is None:
        return {
            "days": int(selected["date"].nunique()),
            "positive_days": None,
            "false_positive_days": None,
            "total_delta_profit": None,
            "avg_delta_profit": None,
            "worst_delta_profit": None,
        }
    delta = true_delta.astype(float)
    return {
        "days": int(selected["date"].nunique()),
        "positive_days": int((delta > 0.0).sum()),
        "false_positive_days": int((delta <= 0.0).sum()),
        "total_delta_profit": float(delta.sum()),
        "avg_delta_profit": float(delta.mean()),
        "worst_delta_profit": float(delta.min()),
    }


def replay_folds(
    scored_dir: str | Path,
    source_name: str,
    proba_threshold: float,
    min_expected_delta: float,
    require_baseline_stability: bool,
    max_abs_charge_delta: int,
    direction: str,
    recent_centered_min: float | None,
    hist_centered_min: float | None,
    risk_proba_threshold: float | None,
    min_risk_expected_delta: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for fold, frame in _load_scored_folds(scored_dir):
        eligible = _base_charge_only_filter(
            frame,
            proba_threshold=proba_threshold,
            min_expected_delta=min_expected_delta,
            require_baseline_stability=require_baseline_stability,
            max_abs_charge_delta=max_abs_charge_delta,
            direction=direction,
            recent_centered_min=recent_centered_min,
            hist_centered_min=hist_centered_min,
            risk_proba_threshold=risk_proba_threshold,
            min_risk_expected_delta=min_risk_expected_delta,
        )
        selected = _select_one_per_day(eligible)
        if not selected.empty:
            selected.insert(0, "source", source_name)
            selected.insert(1, "fold", int(fold))
            detail_rows.append(selected)
        row = _summarize_selected(selected)
        row.update({"source": source_name, "fold": int(fold)})
        summary_rows.append(row)
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    total = _summarize_selected(detail) if not detail.empty else _summarize_selected(pd.DataFrame())
    total.update({"source": source_name, "fold": "all"})
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    return detail, summary


def preview_test(
    scored_windows: str | Path,
    proba_threshold: float,
    min_expected_delta: float,
    require_baseline_stability: bool,
    max_abs_charge_delta: int,
    direction: str,
    recent_centered_min: float | None,
    hist_centered_min: float | None,
    risk_proba_threshold: float | None,
    min_risk_expected_delta: float | None,
) -> pd.DataFrame:
    frame = pd.read_csv(scored_windows)
    frame["date"] = frame["date"].astype(str)
    eligible = _base_charge_only_filter(
        frame,
        proba_threshold=proba_threshold,
        min_expected_delta=min_expected_delta,
        require_baseline_stability=require_baseline_stability,
        max_abs_charge_delta=max_abs_charge_delta,
        direction=direction,
        recent_centered_min=recent_centered_min,
        hist_centered_min=hist_centered_min,
        risk_proba_threshold=risk_proba_threshold,
        min_risk_expected_delta=min_risk_expected_delta,
    )
    return _select_one_per_day(eligible)


def _write_csv(frame: pd.DataFrame, path: str) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay conservative charge-only replacement rules on scored windows."
    )
    parser.add_argument("--scored-dir", required=True)
    parser.add_argument("--source-name", default="charge_only")
    parser.add_argument("--proba-threshold", type=float, default=0.40)
    parser.add_argument("--min-expected-delta", type=float, default=-100000.0)
    parser.add_argument("--require-baseline-stability", action="store_true")
    parser.add_argument("--max-abs-charge-delta", type=int, default=1)
    parser.add_argument("--direction", choices=["earlier", "later", "any"], default="earlier")
    parser.add_argument("--recent-centered-min", type=float, default=None)
    parser.add_argument("--hist-centered-min", type=float, default=None)
    parser.add_argument("--risk-proba-threshold", type=float, default=None)
    parser.add_argument("--min-risk-expected-delta", type=float, default=None)
    parser.add_argument("--detail-output", default="")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--test-window-input", default="")
    parser.add_argument("--test-preview-output", default="")
    args = parser.parse_args()

    detail, summary = replay_folds(
        args.scored_dir,
        source_name=args.source_name,
        proba_threshold=args.proba_threshold,
        min_expected_delta=args.min_expected_delta,
        require_baseline_stability=args.require_baseline_stability,
        max_abs_charge_delta=args.max_abs_charge_delta,
        direction=args.direction,
        recent_centered_min=args.recent_centered_min,
        hist_centered_min=args.hist_centered_min,
        risk_proba_threshold=args.risk_proba_threshold,
        min_risk_expected_delta=args.min_risk_expected_delta,
    )
    _write_csv(detail, args.detail_output)
    _write_csv(summary, args.summary_output)
    print("charge_only_replay_summary=" + ", ".join(f"{k}={v}" for k, v in summary.iloc[-1].items()))

    if args.test_window_input:
        preview = preview_test(
            args.test_window_input,
            proba_threshold=args.proba_threshold,
            min_expected_delta=args.min_expected_delta,
            require_baseline_stability=args.require_baseline_stability,
            max_abs_charge_delta=args.max_abs_charge_delta,
            direction=args.direction,
            recent_centered_min=args.recent_centered_min,
            hist_centered_min=args.hist_centered_min,
            risk_proba_threshold=args.risk_proba_threshold,
            min_risk_expected_delta=args.min_risk_expected_delta,
        )
        _write_csv(preview, args.test_preview_output)
        print(f"charge_only_test_preview_days={preview['date'].nunique() if not preview.empty else 0}")


if __name__ == "__main__":
    main()
